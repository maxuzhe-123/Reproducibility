# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
import os
import re

from .repo_utils import Tools, FilePathBuilder, CodexTokenizer, CodeGenTokenizer, CONSTANTS


class PromptBuilder:
    def __init__(self, query_lines_with_retrieval_results, task_path, log_message, tokenizer):
        self.query_lines_with_retrieval_results = query_lines_with_retrieval_results
        self.log_message = log_message
        if tokenizer == CodexTokenizer:
            self.tokenizer = CodexTokenizer()
            self.max_retrieval_length = 2000  # half of the max length of the model
        elif tokenizer == CodeGenTokenizer:
            self.tokenizer = CodeGenTokenizer()
            self.max_retrieval_length = 1000
        tasks = Tools.load_jsonl(task_path)
        self.tasks_by_task_id = {task['metadata']['task_id']: task for task in tasks}
        self.seperator = '# ' + '-' * 50
        self.max_examples = 10  # maximum number of examples to be included in the prompt

    def _make_a_block(self, retrieved_context):
        content, sim_score = retrieved_context
        metadata = content['metadata']
        # put the file path in the comment
        assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']
        f_paths = ['/'.join(x['fpath_tuple'][1:]) for x in metadata]
        f_paths_str = '\n'.join([f'# {f_path}' for f_path in f_paths])
        f_path_comment = f'# the below code fragment can be found in:'
        # put code lines in the comment
        content_lines = content['context'].splitlines()
        content_lines_comment = [f'# {line}' for line in content_lines]
        # aggregate the comment and the code lines

        block_str = '\n'.join(
            [f_path_comment, f_paths_str, self.seperator] + content_lines_comment + [self.seperator]) + '\n'
        tokenized_block = self.tokenizer.tokenize(block_str)
        token_len = len(tokenized_block)
        return block_str, token_len

    def _make_an_extended_block(self, retrieved_context):
        content, sim_score = retrieved_context
        metadata = content['metadata']
        # put the file path in the comment
        assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']
        f_paths = ['/'.join(x['fpath_tuple'][1:]) for x in metadata]
        f_paths_str = '\n'.join([f'# {f_path}' for f_path in f_paths])
        f_path_comment = f'# the below code fragment can be found in:'
        # put code lines in the comment
        original_code = Tools.read_code(os.path.join(FilePathBuilder.repo_base_dir, *metadata[0]['fpath_tuple']))
        code_lines = original_code.splitlines()
        end_line_no = metadata[0]['end_line_no']
        window_size = metadata[0]['window_size']
        slice_size = metadata[0]['slice_size']
        new_end_line_no = min(end_line_no + window_size // slice_size, len(code_lines))
        new_start_line_no = max(0, new_end_line_no - window_size)
        content_lines = code_lines[new_start_line_no:new_end_line_no]
        content_lines_comment = [f'# {line}' for line in content_lines]
        # aggregate the comment and the code lines
        block_str = '\n'.join(
            [f_path_comment, f_paths_str, self.seperator] + content_lines_comment + [self.seperator]) + '\n'
        tokenized_block = self.tokenizer.tokenize(block_str)
        token_len = len(tokenized_block)
        return block_str, token_len

    def _build_prompt(self, mode, prompt, top_k_context):
        prepend_context = "# Here are some relevant code fragments from other files of the repo:\n"
        prepend_context += self.seperator + '\n'
        current_token_length = 20  # the length of the head_prompt, same for codex and codegen tokenizer
        prepend_blocks = []
        chosen_context = []
        make_block_func = self._make_an_extended_block if mode == CONSTANTS.rg else self._make_a_block
        for retrieved_context in top_k_context[::-1]:
            if len(chosen_context) >= self.max_examples:
                break
            block_str, token_len = make_block_func(retrieved_context)
            if current_token_length + token_len < self.max_retrieval_length:
                prepend_blocks.insert(0, block_str)
                current_token_length += token_len
                chosen_context.append(retrieved_context)
            else:
                continue
        prepend_context += ''.join(prepend_blocks)  # all the blocks already have a line break at the end
        return prepend_context + '\n' + prompt, chosen_context

    def build_2nd_stage_input_file(self, mode):
        new_prompt_lines = []
        for query_line in self.query_lines_with_retrieval_results:
            task_id = query_line['metadata']['task_id']
            task = self.tasks_by_task_id[task_id]
            old_prompt = task['prompt']
            top_k_context = query_line['top_k_context']
            new_prompt, chosen_context = self._build_prompt(mode, old_prompt, top_k_context)
            new_prompt_line = {
                'prompt': new_prompt,
                'metadata': task['metadata'],
            }
            new_prompt_line['metadata']['query_window'] = {
                'context': query_line['context'],
                'metadata': query_line['metadata'],
            }
            new_prompt_line['metadata']['top_k_context'] = [
                {
                    'context': x[0]['context'],
                    'metadata': x[0]['metadata'],
                    'sim_score': x[1],
                } for x in chosen_context
            ]
            # new_prompt_line['metadata']['window_size'] = query_line['metadata']['window_size']
            # new_prompt_line['metadata']['slice_size'] = chosen_context[0][0]['metadata'][0]['slice_size']
            new_prompt_lines.append(new_prompt_line)
        print('done! ' + self.log_message)
        return new_prompt_lines


class BuildPromptWrapper:
    def __init__(self, vectorizer, benchmark, repos, window_size, slice_size, tokenizer):
        if vectorizer == 'one-gram':
            self.vector_path_builder = FilePathBuilder.one_gram_vector_path
        elif vectorizer == 'ada002':
            self.vector_path_builder = FilePathBuilder.ada002_vector_path
        self.max_top_k = 20
        self.repos = repos
        self.window_size = window_size
        self.slice_size = slice_size
        if benchmark == CONSTANTS.line_benchmark:
            self.task_path = FilePathBuilder.random_line_completion_benchmark
        elif benchmark == CONSTANTS.api_benchmark:
            self.task_path = FilePathBuilder.api_completion_benchmark
        elif benchmark == CONSTANTS.short_api_benchmark:
            self.task_path = FilePathBuilder.short_api_completion_benchmark
        elif benchmark == CONSTANTS.short_line_benchmark:
            self.task_path = FilePathBuilder.short_random_line_completion_benchmark
        elif benchmark == CONSTANTS.codereval_benchmark:
            self.task_path = FilePathBuilder.codereval_benchmark

        self.benchmark = benchmark
        self.tokenizer = tokenizer

    def _run(self, mode, query_window_path_builder, output_file_path):

        BAD = re.compile(r"(?:^|/).*passk_validte.*\.py$|__tmp_eval__", re.IGNORECASE)

        def _meta_path_str(meta):
            if not meta:
                return ""
            fp = meta.get("lsp-path") or meta.get("path") or meta.get("file_path") or meta.get("relpath")
            if not fp:
                fp = meta.get("fpath_tuple")
                if isinstance(fp, list):
                    fp = "/".join(fp)
            return str(fp or "")

        def _is_bad_hit(item):
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            p = _meta_path_str(meta)
            if BAD.search(p):
                return True
            ctx = item.get("context", "") if isinstance(item, dict) else ""
            return bool(BAD.search(ctx))

        def _filter_retrieval(q):
            if not isinstance(q, dict):
                return q
            for key in ("top_k_context", "retrieved", "contexts"):
                if key in q and isinstance(q[key], list):
                    q[key] = [x for x in q[key] if not _is_bad_hit(x)][: self.max_top_k]
            return q

        workers = []
        for repo in self.repos:
            query_window_path = query_window_path_builder(repo, self.window_size)
            query_line_path = self.vector_path_builder(query_window_path)

            repo_window_path = FilePathBuilder.repo_windows_path(repo, self.window_size, self.slice_size)
            repo_embedding_path = self.vector_path_builder(repo_window_path)

            retrieval_results = FilePathBuilder.retrieval_results_path(
                query_line_path, repo_embedding_path, self.max_top_k
            )

            raw = Tools.load_pickle(retrieval_results)
            filtered = [_filter_retrieval(q) for q in raw]

            log_message = f"repo: {repo}, window: {self.window_size}, slice: {self.slice_size}"
            workers.append(PromptBuilder(filtered, self.task_path, log_message, self.tokenizer))

        lines = []
        for worker in workers:
            lines += worker.build_2nd_stage_input_file(mode)

        # ===== leakage fix: never write ground_truth into prompt jsonl =====
        for it in lines:
            meta = it.get("metadata") if isinstance(it, dict) else None
            if isinstance(meta, dict):
                meta.pop("ground_truth", None)

        Tools.dump_jsonl(lines, output_file_path)

    def build_first_search_prompt(self, mode, output_path):
        query_line_path_temp = functools.partial(FilePathBuilder.search_first_window_path, self.benchmark, mode)
        self._run(mode, query_line_path_temp, output_path)

    def build_prediction_prompt(self, mode, prediction_path, output_path):
        query_line_path_temp = functools.partial(FilePathBuilder.gen_first_window_path, self.benchmark, mode,
                                                 prediction_path)
        self._run(mode, query_line_path_temp, output_path)

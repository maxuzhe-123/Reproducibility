# import multiprocessing as mp
# def _exec(code, test, ret):
#     glb = {"__builtins__": __builtins__}; lcl = {}
#     try:
#         exec(code, glb, lcl); exec(test, glb, lcl)
#         ret["ok"]=True; ret["err"]=""
#     except Exception as e:
#         ret["ok"]=False; ret["err"]=f"{type(e).__name__}: {e}"
# def run_with_timeout(code, test, timeout_sec=6):
#     m=mp.Manager(); ret=m.dict(); p=mp.Process(target=_exec,args=(code,test,ret))
#     p.start(); p.join(timeout=timeout_sec)
#     if p.is_alive(): p.terminate(); p.join(); return False,"Timeout"
#     return ret.get("ok",False), ret.get("err","Unknown")

# import multiprocessing as mp
# import traceback
#
# def _exec(code, test, ret):
#     # Use a single namespace as both globals and locals
#     ns = {"__builtins__": __builtins__}
#     try:
#         exec(code, ns, ns)   # function definitions + imports go into ns
#         exec(test, ns, ns)   # tests are also executed in the same ns
#         ret["ok"] = True
#         ret["err"] = ""
#     except Exception:
#         # Return the full traceback to see which assert failed
#         ret["ok"] = False
#         ret["err"] = traceback.format_exc()
#
# def run_with_timeout(code, test, timeout_sec=6):
#     m = mp.Manager()
#     ret = m.dict()
#     p = mp.Process(target=_exec, args=(code, test, ret))
#     p.start()
#     p.join(timeout=timeout_sec)
#     if p.is_alive():
#         p.terminate()
#         p.join()
#         return False, "Timeout"
#     return ret.get("ok", False), ret.get("err", "Unknown")

# if __name__ == "__main__":
#     code = "def find_lists(lst):\n    return len(lst)"
#     test = """assert find_lists(([1, 2, 3, 4], [5, 6, 7, 8])) == 2
# assert find_lists(([1, 2], [3, 4], [5, 6]))  == 3
# assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1])) == 1
# """
#
#     ns = {"__builtins__": __builtins__}
#     exec(code, ns, ns)
#     exec(test, ns, ns)
#     print("all good")

# exec_code.py
import multiprocessing as mp
import signal
import traceback
import re
import textwrap
import typing as _t

# ================== Code extraction and cleaning functions ==================
# def extract_python_function(raw: str):
#     """
#     Clean the model output:
#     - remove <think> ... </think> or a standalone <think>
#     - remove explanatory natural language
#     - extract the first function block starting with "def xxx("
#     - remove a trailing 'Stop.' or similar marker
#     """
#     text = raw.strip()
#
#     # (1) Remove <think> blocks
#     if "<think>" in text:
#         # Remove <think> ... </think>
#         text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#         # Remove a possibly unclosed <think> and the content before it
#         text = text.split("<think>")[-1].strip()
#
#     # (2) Locate the first function definition starting with def
#     m = re.search(r"^\s*def\s+[a-zA-Z_]\w*\s*\(", text, re.M)
#     if not m:
#         # If no def is found, return the raw text directly (it will fail at the exec stage)
#         return text
#     start = m.start()
#     text = text[start:]
#
#     # (3) Remove the trailing Stop.
#     text = text.replace("Stop.", "").strip()
#
#     return text

def _infer_func_name_from_code(code: str):
    """Guess the main function name from the generated code (HumanEval/MBPP are single-function tasks)"""
    m = re.search(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", code, re.M)
    return m.group(1) if m else None


def _pick_any_callable(ns):
    """Fallback: pick the callable in the namespace that most resembles the main function"""
    blacklist = {
        '__builtins__', 'check', 'print', 'input', 'exec', 'eval',
    }
    candidates = []
    for k, v in ns.items():
        if k in blacklist or k.startswith('_'):
            continue
        if callable(v):
            candidates.append((k, v))
    candidates.sort(key=lambda kv: (len(kv[0]), kv[0]))
    return candidates[0][1] if candidates else None


def _exec(code, test, ret, entry_point=None):
    # Unified namespace
    ns = {
        "__builtins__": __builtins__,
        # Add common type aliases to prevent NameError for List etc. in type hints
        "List": _t.List,
        "Dict": _t.Dict,
        "Tuple": _t.Tuple,
        "Set": _t.Set,
        "Optional": _t.Optional,
    }
    ns["print"] = lambda *a, **k: None

    try:
        code_clean = textwrap.dedent(code)
        test_clean = textwrap.dedent(test)

        # 1) Execute the candidate code first
        exec(code_clean, ns, ns)

        # 2) Then execute the test code
        exec(test_clean, ns, ns)

        # 3) If the tests define check(candidate), call it automatically
        if "check" in ns and callable(ns["check"]):
            cand = None

            # ✅ Preferred: use the entry_point provided by HumanEval
            if entry_point is not None and entry_point in ns and callable(ns[entry_point]):
                cand = ns[entry_point]
            else:
                if "check" in ns and callable(ns["check"]):
                    # ✅ HumanEval strict: entry_point must be used
                    if entry_point is None:
                        raise ValueError("Missing entry_point for HumanEval.")
                    if entry_point not in ns or not callable(ns[entry_point]):
                        raise NameError(f"entry_point '{entry_point}' not found in generated code.")
                    ns["check"](ns[entry_point])
                # # Second choice: guess from the code with a regex
                # func_name = _infer_func_name_from_code(code_clean)
                # if func_name and func_name in ns and callable(ns[func_name]):
                #     cand = ns[func_name]
                # else:
                #     # Last resort: pick anything that looks like the main function
                #     cand = _pick_any_callable(ns)

            if cand is None:
                raise NameError(
                    "无法找到待测的候选函数（candidate）。"
                    "请检查模型输出是否包含正确的 def 函数定义。"
                )

            ns["check"](cand)

        # If the tests are top-level asserts, nothing needs to be done here;
        # the earlier exec(test_clean, ns, ns) has already run them.

        ret["ok"] = True
        ret["err"] = ""

    except Exception:
        ret["ok"] = False
        ret["err"] = traceback.format_exc()


# timeout handler
def _timeout_handler(signum, frame):
    raise TimeoutError()

def run_with_timeout(code, test, timeout_sec=6, entry_point=None):
    """
    Safe timeout via signal.alarm (works on both macOS and Linux).
    multiprocessing is no longer used (spawn caused timeouts).
    """
    # Register the alarm handler
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)

    ret = {"ok": False, "err": ""}

    try:

        # Use the original _exec (unchanged)
        _exec(code, test, ret, entry_point)

        # Turn off the alarm after successful execution
        signal.alarm(0)
        return ret["ok"], ret["err"]

    except TimeoutError:
        # Timeout
        return False, "Timeout"

    except Exception as e:
        # Ordinary exception
        return False, traceback.format_exc()

    finally:
        # Make sure the signal is cleared
        signal.alarm(0)


# def run_with_timeout(code, test, timeout_sec=6, entry_point=None):
#     m = mp.Manager()
#     ret = m.dict()
#     p = mp.Process(target=_exec, args=(code, test, ret, entry_point))
#     p.start()
#     p.join(timeout=timeout_sec)
#     if p.is_alive():
#         p.terminate()
#         p.join()
#         return False, "Timeout"
#     return ret.get("ok", False), ret.get("err", "Unknown")
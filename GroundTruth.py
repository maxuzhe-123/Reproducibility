
import multiprocessing
from pathlib import Path
import traceback, subprocess

import sys
print("RUNNER:", sys.executable)
import tempfile, hashlib, os

# Unified evaluation logic: GT and the three CoderEval experiments share the same PythonExec executor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.codereval_exec import run_codereval_script


PROJECT_ROOT = Path("xxxx/xxxx").resolve()
REPOS_ROOT = (PROJECT_ROOT / "repos").resolve()
project_path = str(REPOS_ROOT) + "/"
project_dir = str(REPOS_ROOT)


class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            self._cconn.send(exception)

    def join(self, timeout):
        super().join(timeout)

        if self.is_alive():
            self.terminate()
        super().join()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
tot_success=0
ic=0
zzz=0
dictt_not_project={}
# dict_std_nonestd={"/LLM Reproducibility/repos/standalone/neo4j-_meta-deprecated.py":"/LLM Reproducibility/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
#                   "/LLM Reproducibility/repos/standalone/neo4j-work-query-unit_of_work.py":"/LLM Reproducibility/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
#                   "/LLM Reproducibility/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py":"/LLM Reproducibility/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"}
dict_std_nonestd = {
    str(REPOS_ROOT / "standalone" / "neo4j-_meta-deprecated.py"):
        str(REPOS_ROOT / "neo4j---neo4j-python-driver" / "src" / "neo4j" / "_meta_deprecated_passk_validte.py"),
    str(REPOS_ROOT / "standalone" / "neo4j-work-query-unit_of_work.py"):
        str(REPOS_ROOT / "neo4j---neo4j-python-driver" / "src" / "neo4j" / "_work" / "query_unit_of_work_passk_validte.py"),
    str(REPOS_ROOT / "standalone" / "krake-krake-controller-kubernetes-hooks-on.py"):
        str(REPOS_ROOT / "rak-n-rok---Krake" / "krake" / "krake" / "controller" / "kubernetes" / "hooks_on_passk_validte.py"),
}
f_map=open("xxxx/xxxx/record_fail_map.json", 'w', encoding="utf-8")
if __name__ == "__main__":
    c=0
    gt_passed = []  # ✅ added here
    gt_failed = []  # ✅ optional
    gt_env_error = []  # Missing environment/dependency errors (ModuleNotFoundError / ImportError) are classified separately
    import sys
    import json

    f = open("xxxx/xxxx/testcasesoriginal", 'r', encoding="utf-8")
    content = f.read()
    f.close()
    content_list = content.split("----------------------------")
    dictt = {}
    for i in range(1, len(content_list)):
        temp_content = content_list[i]
        temp_content_list = temp_content.split("\n")
        _id = temp_content_list[1]
        test_content = ""
        for j in range(2, len(temp_content_list)):
            test_content += temp_content_list[j]
            test_content += "\n"
        dictt[_id] = test_content

    # print(dictt)
    # print(len(dictt))
    f = open(str(PROJECT_ROOT / "src/datasets/CoderEval/CoderEval4Python.json"), 'r', encoding="utf-8")
    content = f.read()
    f.close()

    import os
    import json, dill

    content_json = json.loads(content)
    listtot = []
    dictt_origin = {}
    project_dir = str(REPOS_ROOT)
    for l in content_json['RECORDS']:
        dictt_origin[l["_id"]] = l

    zz = 0
    list_stadnalone_id = []
    for keyy in dictt.keys():
        if dictt[keyy].find(project_path) < 0:
        # if dictt[keyy].find("/LLM Reproducibility/repos/") < 0:
            # zz+=1
            # print("----------------------")
            # print(dictt[keyy])
            if dictt[keyy].find("def ") >= 0:
                list_stadnalone_id.append(keyy)
                zz += 1
    fff = open("xxxx/xxxx/record_fail.txt", 'w', encoding="utf-8")
    ffff = open("xxxx/xxxx/detailfalut.txt", 'w', encoding="utf-8")
    kk = 0
    project_path = str(REPOS_ROOT) + "/"
    for keyy in dictt_origin:
        dictTemp = dictt_origin[keyy]
        if keyy in list_stadnalone_id:
            save_data = project_path + "standalone/" + dictTemp["file_path"].replace(".py", "").replace("/",

                                                                                                        "-") + "-" + \
                        dictTemp["name"] + ".py"
        else:
            file_path = dictTemp['file_path']
            if (REPOS_ROOT / dictTemp["project"].replace("/", "---")).resolve() == (REPOS_ROOT / "neo4j---neo4j-python-driver").resolve():
            # if project_path + dictTemp["project"].replace("/",
            #                                               "---") == "/LLM Reproducibility/repos/neo4j---neo4j-python-driver":
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---") + "/src",
                                         file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            else:
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---"), file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            moz_root = str(REPOS_ROOT / "mozilla---relman-auto-nag")
            if save_data.startswith(moz_root):
                save_data = save_data.replace(moz_root + "/auto-ang", moz_root + "/bugbot")
            # if save_data.find("/LLM Reproducibility/repos/mozilla---relman-auto-nag") >= 0:
            #     save_data = save_data.replace("/LLM Reproducibility/repos/mozilla---relman-auto-nag/auto-ang",
            #                                   "/LLM Reproducibility/repos/mozilla---relman-auto-nag/bugbot")

            templ = save_data.split("/")
            if not os.path.exists(save_data):
                print("notexis!!")
                print(save_data)
                continue
        if save_data in dict_std_nonestd.keys():
            save_data=dict_std_nonestd[save_data]

        # Repo root of the current task (most reliable: use dictTemp["project"])
        repo_name = dictTemp["project"].replace("/", "---")
        repo_root = str((REPOS_ROOT / repo_name).resolve())
        task_id = dictTemp["project"].replace("/", "---") + "/" + dictTemp["_id"]  # ✅ use the dataset _id

        if not os.path.exists(save_data):
            # Record and skip, to avoid "python: can't open file ..."
            ffff.write("filepath:" + save_data + "\n")
            ffff.write("error infoo:\n\nMISSING FILE (standalone path does not exist)\n")
            ffff.write("----------------------------------------------")
            fff.write(save_data + "\n")
            continue

        # Unified evaluation logic: shares run_codereval_script (PythonExec) with the three CoderEval experiments.
        # Same patch/bootstrap, same interpreter (sys.executable), same cwd/PYTHONPATH.
        ok, err = run_codereval_script(
            script_path=save_data,
            repos_root=str(REPOS_ROOT),
            repo_root=repo_root,
            project_root=str(PROJECT_ROOT),
            timeout_sec=30,
            logical_script_path=save_data,
        )

        if ok:
            c += 1
            gt_passed.append(task_id)
        else:
            err = err or ""
            if err.startswith("timeout"):
                zzz += 1
            ffff.write("filepath:"+save_data+"\n")
            ffff.write("error infoo:\n\n" + err + "\n")
            ffff.write("----------------------------------------------")
            ic+=1
            # Distinguish "missing environment/dependency" from "genuine test failure" so that GT can detect environment issues
            if ("ModuleNotFoundError" in err) or ("ImportError" in err):
                gt_env_error.append(task_id)
            else:
                gt_failed.append(task_id)
            if project_path + dictTemp["project"].replace("/", "---") not in dictt_not_project.keys():
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")] = [save_data]
            else:
                dictt_not_project[project_path + dictTemp["project"].replace("/", "---")].append(save_data)
            fff.write(save_data + "\n")
    f_map.write(json.dumps(dictt_not_project))
    f_map.close()
    print(c,ic,zzz)
    Path("gt_passed_task_ids.txt").write_text("\n".join(gt_passed), encoding="utf-8")
    Path("gt_failed_task_ids.txt").write_text("\n".join(gt_failed), encoding="utf-8")  # Optional
    Path("gt_env_error_task_ids.txt").write_text("\n".join(gt_env_error), encoding="utf-8")
    print("saved gt_passed_task_ids.txt", len(gt_passed))
    print(f"passed={len(gt_passed)}  real_failed={len(gt_failed)}  env_error(缺依赖)={len(gt_env_error)}")
    if gt_env_error:
        print("⚠️ 存在环境/依赖缺失任务，请按 detailfalut.txt 里的 ModuleNotFoundError 补装依赖；"
              "见 gt_env_error_task_ids.txt")
    log_path = PROJECT_ROOT / "pythonsol_unittest.log"
    if log_path.exists():
        if log_path.is_dir():
            import shutil

            shutil.rmtree(log_path)
        else:
            log_path.unlink()
    # if os.path.exists("/LLM Reproducibility/pythonsol_unittest.log"):
    #     if os.path.isdir("/LLM Reproducibility/pythonsol_unittest.log"):
    #         os.removedirs("/LLM Reproducibility/pythonsol_unittest.log")
    #     if os.path.isfile("/LLM Reproducibility/pythonsol_unittest.log"):
    #         os.remove("/LLM Reproducibility/pythonsol_unittest.log")


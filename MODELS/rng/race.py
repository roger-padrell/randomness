from test import Net, get_models, test_model
import argparse
from alive_progress import alive_bar
import os
import sys

def clear_term():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For Linux and macOS
    else:
        os.system('clear')
        
def export_txt(cont):
    res = "";
    n = -1;
    for model in cont:
        n += 1;
        mc = cont[model];
        correct = mc["correct"];
        total_n = mc["total_n"];
        props = mc["props"];
        if n != 0:
            res += "\n\n";
        res += f"- {model}\n    - Correct: {correct}/{total_n} ({round(correct/total_n*100, 2)}%)\n    - Proportions:";
        for p in range(len(props)):
            res += f"\n        - {p}: {round(props[p]*100,2)}";
    return res;

def _sorter(item):
    key, value = item
    return 100 - ((value["correct"] / value["total_n"]) * 100);

def sort_models(cont):
    return dict(sorted(cont.items(), key=_sorter))

def _get_medal(n):
    if n == 0:
        medal = "🥇";
    elif n == 1:
        medal = "🥈";
    elif n == 2:
        medal = "🥉";
    else:
        medal = f"({n+1})"
    return medal

def export_md(cont):
    res = "";
    # Sort
    cont = sort_models(cont);
    #Intro
    res += f"# Model Race\n\nNumber of models: {len(cont)}\n\n"
    #ToC
    res += "## Models"
    t = -1;
    for model in cont:
        t += 1;
        res += f"\n- [{_get_medal(t)} {model}](#{_get_medal(t)}-{model})"
        
    n = -1;
    #For each model
    for model in cont:
        n += 1;
        medal = _get_medal(n)
        mc = cont[model];
        correct = mc["correct"];
        total_n = mc["total_n"];
        props = mc["props"];
        res += "\n\n";
        res += f"## {medal} {model}\n### Correct:\n{correct}/{total_n} ({round(correct/total_n*100, 2)}%)\n\n### Proportions:";
        for p in range(len(props)):
            res += f"\n- {p}: {round(props[p]*100,2)}";
    return res;
    
def export(cont, ft):
    if ft == "txt":
        return export_txt(cont);
    elif ft == "json":
        return str(cont);
    elif ft == "md":
        return export_md(cont);
    else:
        print(f"'{ft}' is not a valid export format", file=sys.stderr)

if __name__ == "__main__":
    clear_term()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", nargs="+", required=False, help="List of models")
    parser.add_argument("-s", action="store_true", default=False, help="Silent, hide logs")
    parser.add_argument("-r", type=int, default=10, help="Total runs (default 10)")
    parser.add_argument("-o", type=str, default="txt", help="Export type (md, txt, json)")
    args = parser.parse_args()
    
    # Select models
    if args.m is None:
        print("Select which models to race (by number):", file=sys.stderr)
        all = get_models();
        for m in range(len(all)):
            print(f"    {m}: {all[m]}", file=sys.stderr);
        print("Enter the numbers separed by ',': ", end="", file=sys.stderr)
        selected = input();
        if selected == "":
            models = all;
        else:
            models = [];
            splitsel = selected.split(",");
            for mod in range(len(splitsel)):
                models.append(all[int(splitsel[mod])]);
    else:
        models = args.m;
        
    silent = args.s;
    TOTAL_RUNS = args.r;
    adv = True;
    
    results = {}
    clear_term()
    with alive_bar(len(models)*TOTAL_RUNS, file=sys.stderr) as bar:
        for m in range(len(models)):
            name = models[m].split("/")
            name = name[len(name)-1].split(".")[0]
            if not silent:
                bar.title("Testing " + name)
            results[name] = test_model(models[m], True, adv, TOTAL_RUNS, bar);
    print(export(results, args.o))
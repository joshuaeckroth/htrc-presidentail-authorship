import json
import os
import re
import rich.progress
from guidance import models, select, assistant, user, system
import itertools
import tiktoken
import requests
from collections import Counter

cl100k_base = tiktoken.get_encoding("cl100k_base")

console = rich.get_console()

model = models.LlamaCpp("Phi-3-mini-4k-instruct-fp16.gguf", echo=False)

pres_choices = [
    "George Washington",
    "John Adams",
    "Thomas Jefferson",
    "James Madison",
    "James Monroe",
    "John Quincy Adams",
    "Andrew Jackson",
    "Martin Van Buren",
    "William Henry Harrison",
    "John Tyler",
    "James K. Polk",
    "Zachary Taylor",
    "Millard Fillmore",
    "Franklin Pierce",
    "James Buchanan",
    "Abraham Lincoln",
    "Andrew Johnson",
    "Ulysses S. Grant",
    "Rutherford B. Hayes",
    "James A. Garfield",
    "Chester A. Arthur",
    "Grover Cleveland",
    "Benjamin Harrison",
    "Grover Cleveland",
    "William McKinley",
    "Theodore Roosevelt",
    "William Howard Taft",
    "Woodrow Wilson",
    "Warren G. Harding",
    "Calvin Coolidge",
    "Herbert Hoover",
    "Franklin D. Roosevelt",
    "Harry S. Truman",
    "Dwight D. Eisenhower",
    "John F. Kennedy",
    "Lyndon B. Johnson",
    "Richard Nixon",
    "Gerald Ford",
    "Jimmy Carter",
    "Ronald Reagan",
    "George H. W. Bush",
    "Bill Clinton",
    "George W. Bush",
    "Barack Obama",
    "Donald Trump",
    "Joe Biden"
]

workset_id = "66477ada2600004a07132b23"

if os.path.exists("workset.json"):
    workset = json.load(open("workset.json"))
else:
    workset = requests.get(f"https://data.htrc.illinois.edu/ef-api/worksets/{workset_id}").json()['data']
    with open("workset.json", "w") as f:
        json.dump(workset, f)

if os.path.exists("pres_predictions.json"):
    results = json.load(open("pres_predictions.json"))
else:
    results = {}

with rich.progress.Progress() as progress:
    task = progress.add_task("Presidential Authorship", total=len(workset['htids']))
    for htid in workset['htids']:
        if htid in results and results[htid] is not None:
            progress.update(task, advance=1)
            continue
        try:
            console.rule()
            console.print("HTID: \"%s\"" % htid)
            if os.path.exists(f"volume_{htid}.json"):
                volume = json.load(open(f"volume_{htid}.json"))
            else:
                volume = requests.get(f"https://data.htrc.illinois.edu/ef-api/volumes/{htid}/pages",
                                      params={"seq": ",".join("%08d" % i for i in range(1, 100)), "pos": "false"}).json()['data']
                with open(f"volume_{htid}.json", "w") as f:
                    json.dump(volume, f)
            page_data = volume['pages']
            fulltxt = ""
            for page in page_data:
                if 'body' not in page or page['body'] is None:
                    continue
                if 'tokensCount' not in page['body'] or page['body']['tokensCount'] is None:
                    continue
                fulltxt += " ".join(filter(lambda t: re.search(r'[A-Za-z]+', t), page['body']['tokensCount'].keys()))
            # sliding window of 300 tokens
            tokens = cl100k_base.encode(fulltxt)
            prompts = []
            for i in range(0, len(tokens)-300, 100):
                prompts.append(tokens[i:i+300])
                if len(prompts) == 30:
                    break
            votes = Counter()
            for idx, prompt in enumerate(prompts):
                try:
                    model.reset()
                    console.print("Prompt "+str(idx)+": [white]"+cl100k_base.decode(prompt)+"[/white]")
                    model += "<|user|>Which president wrote these papers?\n\n"+cl100k_base.decode(prompt)+"<|end|>\n<|assistant|>President: \n"
                    lm = model + select(pres_choices, name="pres")
                    votes[lm['pres']] += 1
                    console.print("Guess "+str(idx)+": [yellow]"+lm['pres']+f"[/yellow]")
                except Exception as e:
                    console.print_exception()
            console.print("Votes: "+str(votes))
            # get most frequent guess
            final_pres_choice = votes.most_common(1)[0][0]
            console.print("Final guess: [bold green]"+final_pres_choice+"[/bold green]")
            results[htid] = {"predicted": final_pres_choice, "votes": dict(votes)}
        except Exception as e:
            console.print_exception()
            results[htid] = None
        with open("pres_predictions.json", "w") as f:
            json.dump(results, f)
        progress.update(task, advance=1)



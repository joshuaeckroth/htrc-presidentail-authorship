import json
import os
import rich.progress
from guidance import models, select, assistant, user, system
import itertools
import tiktoken
import requests

cl100k_base = tiktoken.get_encoding("cl100k_base")

console = rich.get_console()

model = models.LlamaCpp("Phi-3-mini-4k-instruct-fp16.gguf", echo=False)

pres_choices = [
    "Washington, George",
    "Adams, John",
    "Jefferson, Thomas",
    "Madison, James",
    "Monroe, James",
    "Adams, John Quincy",
    "Jackson, Andrew",
    "Van Buren, Martin",
    "Harrison, William Henry",
    "Tyler, John",
    "Polk, James K.",
    "Taylor, Zachary",
    "Fillmore, Millard",
    "Pierce, Franklin",
    "Buchanan, James",
    "Lincoln, Abraham",
    "Johnson, Andrew",
    "Grant, Ulysses S.",
    "Hayes, Rutherford B.",
    "Garfield, James A.",
    "Arthur, Chester A.",
    "Cleveland, Grover",
    "Harrison, Benjamin",
    "Cleveland, Grover",
    "McKinley, William",
    "Roosevelt, Theodore",
    "Taft, William Howard",
    "Wilson, Woodrow",
    "Harding, Warren G.",
    "Coolidge, Calvin",
    "Hoover, Herbert",
    "Roosevelt, Franklin D.",
    "Truman, Harry S.",
    "Eisenhower, Dwight D.",
    "Kennedy, John F.",
    "Johnson, Lyndon B.",
    "Nixon, Richard",
    "Ford, Gerald",
    "Carter, Jimmy",
    "Reagan, Ronald",
    "Bush, George H. W.",
    "Clinton, Bill",
    "Bush, George W.",
    "Obama, Barack",
    "Trump, Donald",
    "Biden, Joe"
]

workset_id = "66477ada2600004a07132b23"

workset = requests.get(f"https://data.htrc.illinois.edu/ef-api/worksets/{workset_id}").json()['data']

results = {}
with rich.progress.Progress() as progress:
    task = progress.add_task("Presidential Authorship", total=len(workset['htids']))
    for htid in workset['htids']:
        try:
            console.rule()
            console.print("HTID: \"%s\"" % htid)
            volume = requests.get(f"https://data.htrc.illinois.edu/ef-api/volumes/{htid}/pages",
                                  params={"seq": ",".join("%08d" % i for i in range(1, 20)), "pos": "false"}).json()['data']
            model.reset()
            page_data = volume['pages']
            prompt = ""
            for page in page_data:
                if 'body' not in page or page['body'] is None:
                    continue
                if 'tokensCount' not in page['body'] or page['body']['tokensCount'] is None:
                    continue
                prompt += " ".join(page['body']['tokensCount'].keys())
            prompt = cl100k_base.decode(cl100k_base.encode(prompt)[:300])
            console.print("Input: [white]\"%s\"[/white]" % prompt)
            model += "<|user|>Which president wrote these papers?\n\n"+prompt+"<|end|>\n<|assistant|>President: \n"
            lm = model + select(pres_choices, name="pres")
            console.print("Guess: [yellow]"+lm['pres']+f"[/yellow]")
            results[htid] = lm['pres']
        except Exception as e:
            console.print_exception()
            results[htid] = None
        with open("pres_predictions.json", "w") as f:
            json.dump(results, f)
        progress.update(task, advance=1)



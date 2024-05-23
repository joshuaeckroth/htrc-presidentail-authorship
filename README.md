# HTRC Presidential Authorship
The HTRC Workset "[USA Presidential Papers](https://htrc.github.io/torchlite-handbook/worksets.html)" contains 779 volumes of "collections of the Papers of the US Presidents over time, Hoover to Obama."

These volumes are not always identified with the president's name. Thus, we have built a technique to predict the presidents based on the texts of the volumes.

The HTRC dataset only provides Extracted Features, which is essentially a list of words that appear on each page of each volume. We do not have the original text. Even so, we believe an LLM can examine a list of words and predict the president.

This code fetches each volume (and saves each to a file for caching), then generates a sliding window of 300 tokens over the text. For each 300 tokens, an LLM is asked to predict the president. We use the [guidance](https://github.com/guidance-ai/guidance) library to constrain the LLM's choices to names of presidents (Hoover to Obama). Since the prediction may be somewhat random depending on the quality of the window of text, we run 30 predictions over different regions of text and take the most common prediction (a voting technique). The model's predictions (votes) are also recorded in the output file.

On a 4090 GPU, the code seems to take about 4.5 hours to run through all volumes.

## Running

Install the Python packages in `requirements.txt`.

Download an LLM like [Phi-3-mini-4k-instruct-fp16.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-fp16.gguf?download=true). Put the downloaded file in the same directory as the code. You may use a different model file if you update the code to reflect the change.

Run with: `python pres.py`

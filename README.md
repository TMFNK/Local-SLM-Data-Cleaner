# Local SLM Data Cleaner

Fine-tune a small language model (SLM) on 100% synthetic data to clean messy SAP-style
master data, and run the whole thing locally on a Mac. No client data, no cloud,
no GPU rental. The intelligence lives in a model you own, and the only thing it
costs to run is the electricity!

What does it accomplish? It takes a messy master-data record (vendor, customer,
material, cost center, GL account) and normalizes it to a documented house
convention. That means trimmed text and fixed casing, ISO country and currency
codes, controlled legal-form, unit and status codes, and canonical VAT, IBAN,
phone, date and amount formats. Missing values become `null`.

```jsonc
// in
{ "name1": "  Muster  Handels ", "legalForm": "mbH", "city": "Muenchen ",
  "country": "Germany", "iban": "de89 3704 0044 0532 0130 00",
  "currency": "€", "status": "aktiv", "validFrom": "01.03.2024", "amount": "1.234,56" }
// out
{ "name1": "Muster Handels", "legalForm": "GmbH", "city": "Muenchen",
  "country": "DE", "iban": "DE89370400440532013000",
  "currency": "EUR", "status": "active", "validFrom": "2024-03-01", "amount": 1234.56,
  "confidence": 1.0, "changes": ["country: 'Germany' -> 'DE'", ...] }
```

---

## Why

If you work with sensitive master data, and especially under GDPR / DSGVO,
sending records to a third-party cloud LLM is often not allowed in the first
place. This project shows a way around that problem.

Everything runs on your own hardware. No data leaves the machine, there is no
third-party API in the loop, and there is nothing to subscribe to. It can run
completely offline (air-gapped), so data protection is the starting point rather
than something you add later.

It is also small. The model is 0.6B parameters and takes about 1 GB on an
ordinary 8 GB laptop. Again, no GPU cluster and no per-token bill needed.

The interesting part is what it does. Hand-written pipelines in SQL, dbt,
PySpark or plain Python handle the clean, expected cases well. The trouble is the
long tail: every new spelling, alias, format or encoding is one more rule that
someone has to write and then maintain forever. A small fine-tuned model learns
the general normalization behaviour and handles that long tail in one pass,
sitting next to the deterministic rules that stay exact where they apply.

And it stays yours! It is built on Qwen (Apache-2.0), so the weights are open,
you can inspect them, run them offline, and keep them forever. There is no
vendor to lock you in.

To be clear, this is not a replacement for your data stack. It is a small, local,
private model that works alongside it and picks up the edge cases the rules miss,
without ever exposing your data to anyone.

## Für deutsche Unternehmen, kurz gefasst

Sensible Stammdaten an eine ausländische Cloud-KI zu senden, ist unter der DSGVO
oft keine Option. Dieses Projekt zeigt einen anderen Weg: ein kleines, quelloffenes
Modell (Qwen, Apache-2.0), das vollständig lokal und bei Bedarf komplett vom Netz
getrennt (Air-Gap) läuft, auf einem normalen Laptop, in rund 1 GB. Keine Cloud,
kein Abo, keine laufenden Token-Kosten, keine Datenweitergabe! Es ergänzt
bestehende Pipelines (SQL, dbt, PySpark), indem es genau die unsauberen Sonderfälle
abfängt, die regelbasierter Code regelmäßig übersieht. Datenschutz und
Datensouveränität sind hier der Standard, nicht die Ausnahme.

Beratung und Umsetzung: [mbitai.com](https://www.mbitai.com).

---

## The enterprise version

This repo is the DEMO version: one laptop in the afternoon, and you can watch the whole
idea work end-to-end. For production use there is a bigger sibling,
[Enterprise-SLM-Data-Cleaner](https://github.com/TMFNK/Enterprise-SLM-Data-Cleaner),
which takes the same proven core and adds the layers that a company actually needs
before trusting an AI with its master data:

- **Client-specific conventions as files.** The house standard lives in an
  editable YAML spec per client. A data steward changes the rules, nobody
  rewrites software.
- **An append-only audit trail.** Every cleaning decision is recorded: input,
  output, every single change, confidence, and the exact version (hash) of
  both the model weights and the convention file. Uncertain records go to a
  manual review queue, never silently accepted.
- **Air-gapped delivery.** Everything ships as one container that runs with
  its network stack removed (`--network none`) and refuses to start if the
  model weights do not match the fingerprint pinned in version control.
- **A quality gate on every change.** A pinned adversarial test suite (is
  "Bavaria" wrongly "corrected" to a country? is "mbH" recognized as GmbH?)
  blocks any code or convention change that alters documented behavior.
- **A swappable base model.** The stack is model-agnostic! Companies that
  prefer not to run a Chinese base model can use a European one (Teuken-7B
  from Fraunhofer, EuroLLM from an EU project) or a US model under MIT
  license, with the same pipeline and the same eval gate.

Kurz auf Deutsch: die Enterprise-Version ergänzt dieses Demo-Projekt um
mandantenfähige Konventionsdateien, ein unveränderliches Audit-Protokoll mit
manueller Prüfschlange, einen komplett vom Netz getrennten Container und ein
austauschbares Basismodell (auf Wunsch europäisch). Beratung und Umsetzung:
[mbitai.com](https://www.mbitai.com).

---

## The main idea

Most data cleaning is written by hand: someone codes a rule for every case they
can think of. This project turns that around. We write the rules once, use them to
mass-produce practice examples, and train a small model on those examples until it
can clean records on its own, including messy cases the original rules never
covered. The trained model then runs on your own machine.

That trick has a name: knowledge distillation. A "teacher" that already knows the
task (here, our rule-based algorithm) produces labelled examples, and a small
"student" model learns from them. The student ends up smaller, faster, and more
flexible than the teacher.

```text
   convention_spec.py            the rules, our "teacher"
        |
        |  generate messy records and label the clean answer
        v
   synthetic dataset             thousands of messy -> clean pairs
        |
        |  fine-tune with MLX + LoRA
        v
   Qwen3-0.6B, now specialised   the "student", about 1 GB
        |
        |  export to GGUF, serve with llama.cpp
        v
   clean records, on your Mac    no cloud, works offline
```

Each step below maps onto this picture: Step 5 builds the dataset, Step 7 trains
the student, Step 8 packages it, and Step 9 checks it learned the job.

Want the reasoning behind these choices? See [docs/concepts.md](docs/concepts.md)
for a deeper explainer: why a tiny model is enough, base vs instruct models, what
LoRA actually changes, quantization, and how to read a learning curve.

---

# Complete beginner's guide

Never trained a model before? This walks you through every step. You do not need
to understand machine learning to follow it. Just copy the commands. The whole
thing takes about 30 to 45 minutes, and most of that is waiting for downloads and
training.

Each step below tells you three things: what to type, what the command actually
does, and what you should see when it worked. If a step goes wrong, check the
[Troubleshooting](#troubleshooting) list at the end before retrying.

### What you need first

- A Mac with Apple Silicon (M1/M2/M3/M4, anything from 2020 on). 8 GB RAM is enough.
- About 5 GB of free disk space, plus an internet connection for the one-time downloads.
- 30 to 45 minutes.

Not sure whether your Mac has Apple Silicon? Click the Apple menu (top left),
choose "About This Mac", and look at the Chip line. "Apple M1" or later means you
are good. If it says "Intel", the training step will not work, because it relies
on Apple's MLX framework.

### A 30-second glossary explanation

| Term                  | Plain meaning                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| LLM / model           | The "brain": a file that turns input text into output text.                                            |
| Base / instruct model | We use`Qwen3-0.6B`, a small model (0.6 billion parameters).                                            |
| Parameters            | The model's internal numbers. "0.6B" means 600 million of them. More usually means smarter but bigger. |
| Fine-tune             | Teach an existing model your specific task by showing it examples.                                     |
| LoRA                  | A cheap, fast way to fine-tune that runs on a laptop.                                                  |
| Adapter               | The small file LoRA produces: what the model learned, kept separate from the model.                    |
| Synthetic data        | Fake but realistic examples we generate ourselves, no real data.                                       |
| Algorithm             | Our rule-based answer key that says what the clean output should be.                                   |
| Loss                  | The number training prints. Roughly "how wrong the model still is". Down is good.                      |
| Quantization          | Shrinking a model by storing its numbers at lower precision, so it needs less memory.                  |
| GGUF                  | The file format that lets`llama.cpp` run a model efficiently.                                          |
| MLX                   | Apple's tool that does the training on your Mac's chip.                                                |
| llama.cpp             | The tool that runs (serves) the finished model.                                                        |
| Server                | A program that loads the model once, keeps it in memory, and answers requests.                         |
| Hugging Face          | The site models are downloaded from. Think "GitHub for models".                                        |
| Terminal              | The black text app where you type commands (see Step 0).                                               |

---

## Step 0. Open the Terminal

Press `Cmd + Space`, type Terminal, press Enter. A window opens where you type
commands. For every step below, paste the command and press Enter. When a command
finishes it gives you a fresh prompt (a line ending in `%` or `$`).

Two things worth knowing before you start. First, the Terminal shows no feedback
while you paste a password (nothing appears, not even dots). Type it blind and
press Enter. Second, a command has not finished until the prompt comes back. Some
of the commands below print a lot of text while they work; that is normal, let
them run.

> Some steps start a server that keeps running and does not hand you a prompt back.
> That is normal. Leave it running and open a second Terminal window (`Cmd + N`)
> for the next command. Steps 6 and 9 tell you when.

## Step 1. Install the basic tools (one time)

First install Homebrew, the Mac software installer. Skip this if you already have
it. Paste this and follow its prompts:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Homebrew will ask for your Mac login password (that is the blind typing mentioned
above) and then explain what it is about to install. Press Enter to accept. It
can take a few minutes. At the end it may print a short "Next steps" block asking
you to run two lines starting with `echo` and `eval`; if it does, copy and run
them, otherwise the `brew` command will not be found.

Then install Python, git, and the model runner:

```bash
brew install python git llama.cpp
```

This gives you the three tools the project needs: Python runs the scripts, git
downloads the project, and llama.cpp runs the finished model. To check it worked,
run `brew --version` and `git --version`. Each should print a version number
instead of "command not found".

## Step 2. Get this project

```bash
git clone https://github.com/TMFNK/Local-SLM-Data-Cleaner.git
cd Local-SLM-Data-Cleaner
```

The first line downloads the project into a new folder named
`Local-SLM-Data-Cleaner` inside whatever folder your Terminal is currently in
(usually your home folder). The second line moves your Terminal into it.

Everything from here on runs inside this folder. You can see the whole menu of
commands any time with:

```bash
make help
```

`make` is a small command runner. The project defines every step as a short
`make` command so you never have to type the long versions by hand. `make help`
lists them all with the step numbers used in this guide.

## Step 3. Install the Python pieces

```bash
make setup
```

This installs the Python libraries and mlx-lm (the trainer). It also tries to
install `llama.cpp` via Homebrew in case you skipped Step 1.

Expect a minute or two of package names scrolling by. Lines saying "Requirement
already satisfied" are fine; they mean a library was already there. The step
succeeded when it ends with `>> Done. Next: make model` and no red `ERROR` lines
above it.

## Step 4. Download the model, fresh

```bash
make model
```

This downloads `Qwen3-0.6B` (about 1.2 GB) from Hugging Face the first time. You
do not need an account or a login, because the model is open (Apache-2.0). It
caches on your disk, so later steps do not download it again.

What you are downloading is the model's weights: the 600 million numbers that
make up its "brain". They land in a hidden cache folder
(`~/.cache/huggingface`), not in the project folder, which is why you never see
the file directly. A progress bar runs while it downloads, and the step is done
when it prints `model ready`.

## Step 5. Make the training data

```bash
make data
```

This generates about 1,000 fake "messy to clean" record pairs (roughly 1.5 MB) and
splits them into `train`, `valid` and `test` files under `data/`. Want more? Run
`make data N=2000`.

Behind the scenes, the generator does three things for every example: it invents
a perfectly clean record (a fake vendor, customer, material, cost center or GL
account), deliberately messes it up the way real data gets messy (extra spaces,
`Germany` instead of `DE`, `1.234,56` instead of `1234.56`, dates the wrong way
round), and then asks the rule-based algorithm for the correct cleaned version.
The messy record becomes the exercise, the algorithm's answer becomes the
solution. No real company data is involved at any point; every name, IBAN and
VAT number is invented.

The output looks like this (the split is 80% train, 10% valid, 10% test):

```text
  test :   100 -> data/test.jsonl
  valid:   100 -> data/valid.jsonl
  train:   800 -> data/train.jsonl
done. seed=0, total=1000
```

The three files matter for different reasons: `train.jsonl` is what the model
learns from, `valid.jsonl` lets the trainer check itself during training, and
`test.jsonl` is kept aside so we can score the model later on examples it has
never seen.

Then confirm the data is correct:

```bash
make sanity
```

You should see numbers around 100%. This scores the held-out test split against
the rule-based answer key (every example was already checked once at generation
time), so if it says 100% the data is good:

```text
mode           : algorithm (sanity)
examples       : 100
valid JSON     : 100.0%
exact record   : 100.0%
field accuracy : 100.0%
```

This is not the model being tested (there is no model involved yet). It is a
self-consistency check: the answer key agrees with itself, so the exercises we
are about to train on have correct solutions.

## Step 6. Measure the model BEFORE training

This shows how the untrained model does, which is what lets you prove training
helped later.

Why bother measuring first? Because "my model scores 85%" means nothing on its
own. If the stock model already scored 80% before you did anything, your training
added little. If it scored 40%, your training did real work. Science needs a
before and an after, and this is the before.

In your first Terminal, start the stock model. This downloads it fresh and keeps
running:

```bash
make baseline-serve
```

This starts a server: a program that loads the model into memory once and then
sits there waiting to answer requests. That is why it does not give you your
prompt back. It is not stuck, it is working.

Leave that running, and wait until it prints a line saying it is listening on
`http://127.0.0.1:8080`. The first run downloads about 600 MB, so give it a minute.
Do not move on until you see that line.

Now open a second Terminal (`Cmd + N`). Make sure it is in the project folder. If
the new terminal started somewhere else, `cd` into the folder you cloned first (for
example `cd ~/Local-SLM-Data-Cleaner`). Then score the model:

```bash
make baseline
```

If you see "Cannot reach the model server", the server in the first Terminal is
not ready yet. Wait for its "listening" line and run `make baseline` again.

This sends all 100 test records to the model, one at a time, and compares each
answer to the answer key. It takes a few minutes because the untrained model is
being asked to do something it was never taught. The report has the same shape as
the sanity check, but the numbers will be visibly lower. That is the point.

Write down the "field accuracy" number. That is your before. Then go back to the
first Terminal and press `Ctrl + C` to stop the server, which frees memory for
training.

## Step 7. Fine-tune the model

```bash
make train
```

This is the actual training. It prints a loss number that ticks down for a few
minutes. When it finishes you have an `adapters/` folder, which holds what the
model learned.

Here is what is happening while you wait. The trainer shows the model batches of
your messy records, asks it to produce the clean version, measures how far off it
was, and nudges the model's numbers slightly in the direction that would have
made it less wrong. Then it repeats that, 1,000 times. The "loss" it prints is
the how-far-off measurement, so you want to watch it fall: it starts high, drops
quickly in the first stretch, then improves more slowly. Exact values do not
matter, the downward trend does. Every so often it also prints a "val loss",
which is the same measurement on the validation examples the model is not
training on; that number falling too tells you the model is learning the task
and not just memorising.

On an 8 GB Mac, close memory-hungry apps first (browsers with many tabs count).
If training gets killed or the Mac crawls, run `make train BATCH=2`, which uses
less memory per step and simply takes a little longer.

The result, `adapters/`, is small (a few MB). It is not a new model; it is a
compact "diff" of what changed, sitting on top of the unchanged base model. The
next step merges the two.

## Step 8. Package your model into a runnable file

```bash
make fuse
make gguf
```

`fuse` merges what it learned back into the model. `gguf` converts it into the
format `llama.cpp` runs. `make gguf` needs a copy of the llama.cpp source. If you
do not have one, grab it once:

```bash
cd .. && git clone https://github.com/ggml-org/llama.cpp && cd Local-SLM-Data-Cleaner
```

The Makefile expects it at `../llama.cpp`. Point it elsewhere with
`make gguf LLAMA_CPP=/your/path`.

In plain terms: after `fuse` you have a `fused/` folder holding a complete model
with your training baked in. `make gguf` then does two conversions in a row.
First it repacks that folder into a single `.gguf` file (large, roughly the size
of the original download), then it compresses it into the file you will actually
use, `qwen3-0.6b-cleaner-q8_0.gguf` (around 600 MB). The compression step is the
quantization from the glossary: storing the model's numbers with less precision
to halve the size, at a quality cost too small to matter here. When both commands
have finished you can see the files with `ls *.gguf`.

## Step 9. Measure the model AFTER training

In your first Terminal, serve your fine-tuned model. It keeps running:

```bash
make serve
```

This is the same kind of server as in Step 6, but now it loads your own file,
the fine-tuned `qwen3-0.6b-cleaner-q8_0.gguf`, instead of downloading the stock
model. Nothing is downloaded this time; it should print its "listening" line
within seconds.

In the second Terminal, score it:

```bash
make eval
```

Compare this "field accuracy" to your before number from Step 6. The gap between
them is your fine-tune paying off.

The same 100 held-out test records are used, so the comparison is fair: same
questions, same scoring, different model. See "How to read your eval numbers"
below for what each of the three numbers means and what counts as a good result.

## Step 10. Clean a record for real

With the server still running:

```bash
make demo
```

It sends one messy record to your model and prints the cleaned JSON.

Two parts of the output are worth a look. The first line says
`source=model needs_review=False`: the model's answer passed all the rule checks,
so no human needs to look at it. And inside the JSON, the `changes` list names
every single edit the model made (`"country: 'Germany' to 'DE'"` and so on), so
a data steward can audit what happened to the record at a glance.

This demo is also the shape of real usage: your own code would call the model
the same way, one JSON record in, one cleaned JSON record out, with the
rule-based algorithm double-checking every answer. That runtime logic lives in
[`clean.py`](clean.py) and it is deliberately short.

---

## Troubleshooting

- `command not found: make`, `brew` or `git`: redo Step 1 (`brew install ...`).
- `command not found: mlx_lm...`: run `make setup` again.
- `Address already in use` on port 8080: a server is still running in another
  Terminal. Find it and press `Ctrl + C`, or use another port with
  `make serve PORT=8081` (then `make eval PORT=8081`).
- The download is slow: the pulls are one-time (about 1.2 GB for the trainer model
  in Step 4, about 600 MB for the baseline in Step 6) and cache after.
- Out of memory during `make train`: stop any running server first, and if needed
  lower the batch size with `make train BATCH=2`.
- Connection refused or `Cannot reach the model server` or `[Errno 61]` on port
  8080: the model server is not running yet. Make sure `make baseline-serve`
  (Step 6) or `make serve` (Step 9) is running in the other Terminal and has
  printed its "listening" line, then run the command again. The first start also
  has to finish downloading the model before it listens.
- You closed the Terminal mid-way: nothing is lost. Open a new one,
  `cd Local-SLM-Data-Cleaner`, and continue from the step you were on. Finished
  steps do not need to be redone; downloads and generated files are still there.

---

# How it works

The convention is defined in code in [`convention_spec.py`](convention_spec.py).
Its `normalize_record()` function is a deterministic algorithm: it computes the
correct clean output from any messy input. That one function gets us three things
for free:

1. Unlimited perfect labels. The generator corrupts a clean record, then labels it
   with the algorithm, so there is no expensive "teacher" model needed to build the
   dataset.
2. A ground truth for eval. We score the model against the same algorithm.
3. A safety net at runtime. If the model's output fails validation, we fall back
   to the algorithm.

So why use an LLM at all when we have the rules? Because the algorithm only covers
the rules we wrote. The model learns those rules and, on top of that, generalizes
to messiness the rules do not explicitly cover, like novel typos, unseen aliases
and fuzzy matches, and it does the whole record in one pass. The eval measures
exactly how much it adds beyond the rules.

### What fine-tuning does

The model starts as a general instruct model that can hold a conversation about
anything. Fine-tuning shows it thousands of messy-to-clean pairs and gently nudges
its numbers until it reliably produces the clean version for this one task. LoRA
makes that cheap: instead of changing all 600 million parameters, it trains a small
set of add-on numbers and leaves the rest frozen. The model is not memorising the
examples. It is learning the pattern, which is why it can clean records it never
saw during training.

### How to read your eval numbers

`make eval` prints three numbers:

- valid JSON: how often the output was parseable JSON at all. Because we constrain
  the model to the record's schema, this stays at 100%.
- field accuracy: the share of individual fields that match the correct answer.
  This is the main number to watch.
- exact record: how often every field in a record is right at the same time. It is
  stricter, so it always sits below field accuracy.

A good result is your fine-tuned model scoring well above the untrained baseline
from Step 6, and landing close to the rule-based algorithm on the cases the rules
cover. Do not expect a flat 100%. The real value is that the model also handles the
messy long tail the rules never anticipated.

### Project layout

```text
convention_spec.py   the convention + deterministic algorithm (single source of truth)
synth/generate.py    clean-record generator + corruptor -> train/valid/test JSONL
train/               notes on the MLX fine-tuning commands
eval/evaluate.py     valid-JSON rate, exact-record match, field accuracy
clean.py             v1 runtime: model -> validate -> algorithm safety net
Makefile             every step above, as `make <command>`
```

### Model details

We use Qwen3-0.6B (instruct), LoRA fine-tuned with Apple MLX, exported to GGUF and
served by [llama.cpp](https://github.com/ggml-org/llama.cpp). It runs in about 1 GB
on an 8 GB Mac. The output is grammar-constrained to the record's JSON schema, so
it is always valid JSON.

For the why behind all of this (tiny models, base vs instruct, LoRA, quantization,
grammar-constrained decoding), see [docs/concepts.md](docs/concepts.md).

## Roadmap

- normalize a record to the convention (this repo).
- duplicate detection and golden-record merge.
- map arbitrary messy nested JSON onto the target schema.

## About

Built by [mbitai](https://www.mbitai.com), freelance data and AI engineering for
German businesses, with a focus on practical, privacy-first machine learning that
runs where your data already lives. This repo is part of that portfolio and a
worked example: local, tiny, open, and GDPR-friendly by design.

## License

AGPL-3.0 (see [LICENSE](LICENSE)). All sample data is synthetic and invented.

For commercial licensing without AGPL obligations, or help applying this to your own master data, contact [www.mbitai.com](https://www.mbitai.com).

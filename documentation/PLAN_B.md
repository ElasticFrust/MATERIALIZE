# PLAN B — the proof-of-concept site

> **Split out of the approved plan of 2026-09-15**, which was written as one file only because plan
> mode permits editing one. **Plan A and Plan B are standalone and independent.** Plan A is
> `Phase 5/m2/PLAN_A.md`. Nothing in B depends on A except **B2**, which is marked throughout.
>
> **Status 2026-09-16: B1 is BUILT and gated locally; only the deploy remains.** The clone lives
> outside this repository and carries the bundle (`model_v2`, `model_v3`, `meshing`, `inference`,
> a sanitised `checkpoint.pt`), the site layer (`guards`, `render`, `library`, `ui`), the Space
> wrapper, and two gate suites: `test_app.py` (6 gates, no gradio, runs in the research
> environment) and `test_ui.py` (4 gates, needs gradio). All ten pass. **B2 is not started.**
>
> This document describes a deliverable that **deliberately does not live in this repository**. The
> site is a **clone** carrying copied files only, with fresh history and no remote here (B.7). This
> file is MATERIALIZE's record of what that clone is and why.

---

## Context

**What and why.** A public demonstration of the metamaterial forward map: pick or build a periodic
spring network, change its stiffnesses, drag its nodes, and see the effective elastic response — ν(θ)
and E(θ) — respond. Later, state a target and let the page design a network for it.

**It runs the GNN surrogate, never the exact solver.** That is a requirement, not a compromise: the
deployed clone contains no solver, no Phase 3 and no oracle, so no unpublished method leaves this
machine. Responses are labelled as surrogate output; the site does not claim ground truth.

**Independent of Plan A.** B1 shares only a frozen checkpoint and can be built at any time.
**B2 (gradient design) is the sole dependency** — it needs A0.1's torch port to exist and A0.2/A0.3 to
say the gradient is usable. *(Both now hold: A0 is complete, the cosines met their thresholds, and the
verdict is "usable inside a trust region gated on `max|W|`" — see `Phase 5/m2/PLAN_A.md`.)*

**Scope discipline:** proof of concept, demo only, with occasional fixes expected later. Not a
product, not a result.

## B.0 Requirements *(from the user, 2026-09-15)*

| requirement | decision |
|---|---|
| forward engine | **the GNN**, never the exact solver |
| status | proof of concept |
| audience | **all three** — collaborators, broader scientific readers, and the user's own exploration — so **explanations must be available**, not mandatory: clean default view, per-panel "explain this" disclosure |
| interaction | **choose from a library**, **build your own**, **tune stiffnesses** |
| node dragging | **IN SCOPE** for the first build |
| reachability | **must be reachable**; free for now; **the user's own domain** |
| compute location | originally *split static frontend + Python API*; **settled as a single Gradio Space on the user's domain** (see Host, below) |
| inverse side | later, same API contract |

*Note on node dragging:* an earlier draft said it needs the A0.1 torch port. **That is wrong** — for
*forward prediction* a moved mesh only needs `bond_R` rebuilt at fixed `shift` and `prepare()` re-run,
all of which works today in NumPy. The torch port is needed only for *gradient-driven* interaction
("optimise this for me"). Dragging does need the signed-area guard so a drag cannot invert a triangle.

## B.1 The compute decision — server-side Python

Inference is genuinely two calls against the existing checkpoint
(`evaluate_v2.py:105`: `T3.predict(net, T3.prepare(g))`), so the API adapter is thin either way. The
decision is **not** about the model — it is about **who builds the mesh**.

- **"Build your own" needs the Python mesh stack**: `seeds.py` → `scipy.spatial.Delaunay` →
  `mesh_build.build_geometry` / `set_VD`. Reimplementing periodic Delaunay meshing in JS would be a
  **second, unverified implementation of core geometry code** — precisely what this project's DRY and
  verification discipline forbid, and the site would then be free to drift from the science.
- The **ONNX/browser** route additionally requires reimplementing `prepare()` (carriers, triangle
  adjacency, vertex stars, normalisers) in JS, and the model has data-dependent shapes (`n_tri`,
  `n_vert`) that make the export non-trivial to gate.
- The **inverse side, when it lands, needs gradients** → torch → a Python process anyway.
- Latency is a non-issue: a GNN forward on a few hundred triangles is milliseconds; the round trip
  dominates.

**Architecture — split, so hosting and domain are decoupled:**

```
  static frontend  (their own domain; free on Cloudflare Pages / GitHub Pages)
        |  fetch(), CORS
  Python API       (free Python host; FastAPI + torch CPU + the checkpoint)
```

This matches the project's own compute/render separation — the server returns **data**, the browser
renders. Cold starts on free tiers are mitigated by shipping the preset library as precomputed static
JSON, so first paint never waits on the server.

### Fully-static on github.io — considered and REJECTED *(2026-09-15)*

More was achievable statically than first assumed: the preset library, **k tuning**, and **node
dragging** all run client-side, because dragging needs only `bond_R` recomputed at fixed `shift` and
`prepare()` re-run — **connectivity is frozen, so no Delaunay is involved.**

Rejected on five counts, in weight order:

1. **The checkpoint would ship to every visitor.** Every other security concern on the user's list is
   revisable; publishing the weights is not. Static publishes them on day one.
2. **`prepare()` would need reimplementing in JS.** That code has three documented subtle bugs in its
   history — the single-sample `tri_batch` one silently ran a *different architecture* at evaluation
   than at training, 7.8e-02 apart. A second implementation is free to drift from the science.
3. **Topology building would need a JS mesher**, rather than the verified free-standing `mesh_build.py`.
4. **B2 gradient design becomes impossible** — ONNX Runtime Web is inference-only, with no autograd;
   reaching gradients would mean porting the model to TensorFlow.js.
5. Server hardening is a known checklist (B.6), not research.

What static *would* have bought — no endpoint, no crafted-payload vector, no DoS surface, no cost — is
real, and is the reason it was weighed rather than dismissed. It loses to (1).

**github.io still hosts the frontend**, under the user's own domain; it simply cannot host the Python.

### Host — SETTLED 2026-09-16: a PROTECTED Hugging Face Space on PRO, Gradio SDK, CPU Basic

The decision took four turns and each one was forced by a measurement, so the trail is kept: the
reasoning is reusable, and three of the dead ends are things a future reader would otherwise retry.

**1. The original choice (free Docker Space) was refuted at step 0.** HF now states that only
*Static* Spaces are free for everyone, and that "Gradio and Docker Spaces run on compute and require
a paid plan to create: PRO for personal accounts". The hardware table still reads **CPU Basic — 2
vCPU, 16 GB, FREE**, so the RAM figure that ruled out Render is intact; what changed is
**eligibility**, not cost-per-hour. *(The plan itself said to verify this rather than trust a
remembered number. It was right to.)*

**2. The free carve-out did not apply.** Free accounts may host 2 **ZeroGPU** Spaces — Gradio SDK
only — but eligibility requires an account **older than 30 days**, and this account is new. So the
free HF route is closed for about a month. Gradio sells no hosting of its own: `gradio.app`'s "deploy
free" *is* Spaces, and `share=True` is a one-week proxy tunnel to your own machine, not a host.

**3. A fully-static page was considered seriously, and is a good design — for a different site.**
Precompute a sweep (φ × ψ × η, with α reused across a fixed geometry) into ~5 MB of JSON, ship no
weights at all, host free on Cloudflare Pages with a custom domain. It gives the library, the
parameter sliders and both response panels, always-on, with nothing to leak. It cannot give
**freehand** — arbitrary geometry and node moves need the model on unswept input — and it cannot give
**B2** at all, since ONNX Runtime Web has no autograd and finite differences would cost ~54 000
forward passes per design. Freehand and B2 were both wanted, so it lost.

**4. PRO clears three blockers with one subscription**, which is why it wins over a VPS or Cloud Run:

| blocker | what PRO gives |
|---|---|
| free accounts cannot create Gradio Spaces (the 30-day rule) | Space creation, immediately |
| a free Space's repo is public, so `checkpoint.pt` would be published | **protected visibility** — source private, running app publicly reachable |
| the site should live on the user's own address | **custom domain**, CNAME to `hf.space` (PRO-only, and needs public or protected visibility) |

Protected visibility is the one that matters most: it deletes an entire mechanism the earlier draft
needed, in which the weights had to live in a **private model repo** pulled at startup with an
`HF_TOKEN` Space secret because the Space's own repo was necessarily public. With a protected repo
the checkpoint simply ships inside it.

**Known and accepted: PRO is not always-on.** CPU Basic still sleeps when idle and wakes in tens of
seconds; running indefinitely needs paid hardware (~$22/month more). Fine for a demo, and the README
says so, so the wake reads as expected rather than as a fault.

**Not ZeroGPU.** The workload is 30–90 ms of CPU on 397 k parameters. With PRO, plain CPU Basic is
allowed and costs nothing per hour, so there is no reason to take GPU hardware whose allocation
latency and per-visitor quotas (2 min/day unauthenticated, 5 min/day free) would make the app both
slower and rationed. No `spaces` package, no `@spaces.GPU`.

**Gradio, not Docker** *(user's call, 2026-09-16)*. PRO permits either. Docker would allow fluid node
dragging and full control of the page; Gradio costs about two days less and needs no frontend. **One
argument made for Gradio was overstated and is corrected here:** rendering the project's matplotlib
conventions server-side is *not* exclusive to Gradio — a FastAPI app can serve the same PNGs. The
real advantage is development time. The cost is that node editing is **two-click** (pick up / put
down) rather than a drag.

The switch stays cheap because everything below the interface is UI-agnostic: `guards.py`,
`render.py`, `library.py`, `inference.py` and `meshing.py` import no Gradio. Moving to Docker later
means writing a frontend and a thin wrapper around functions that already exist and are already
gated.

## B.2 Backend — SUPERSEDED by the Gradio build; kept for the rules it states

*(2026-09-16: the endpoint table below described a REST API behind a separate static frontend. The
build is a **Gradio app**, so there are no public endpoints at all — the handlers in `ui.py` are
called over Gradio's own transport, and `queue(api_open=False)` closes the auto-generated
programmatic API. The two safety rules and the output restriction still hold verbatim and are
implemented in `guards.py` and `inference.directional`; only the transport changed.)*

### the original adapter design

The site lives in its own repository with copied files only (B.6/B.7). It therefore imports nothing
from MATERIALIZE at runtime — no `Phase 2` solver, no `Phase 3`, no `verification_tools`. The only
physics code it carries is `mesh_build.py` and the GNN modules.

| endpoint | returns |
|---|---|
| `GET /api/library` | the preset catalogue (precomputed, also shipped static) |
| `POST /api/mesh` | build a mesh from (family, parameters, seed) → graph JSON |
| `POST /api/predict` | graph + `k` → ν(θ), E(θ), the two scalars, timing — **and nothing else** |
| `POST /api/design` | **B2 only** (B.5) — gradient design through the GNN |

*(Corrected 2026-09-16: the `/api/predict` row used to promise `C6` and a per-triangle ν/E field.
That contradicted B.3 and B.6/9 in the same document — the user's call was to omit both, precisely
because `W` is recoverable from a per-triangle prediction. The restrictive reading wins; the row is
now the narrow one, and the server computes ν(θ)/E(θ) rather than shipping the tensor it came from.)*

**Two hard safety rules** (the full list is B.6):

1. **No solver and no oracle, at all.** Neither is present in the deployed clone. Mesh validity is
   still enforced on every input — `mesh_build.check_mesh_preconditions` plus a signed-area check —
   because an invalid mesh gives the GNN meaningless inputs even when nothing can crash.
2. **Cap the inputs** — node count, triangle count, k-contrast, request timeout — so one request
   cannot wedge the host.

## B.3 Frontend — what is actually shown *(agreed with the user, 2026-09-15)*

**Shown:**

1. **The network, bonds coloured by `k`** — tiled-continuous and cropped (project convention): viridis
   by `k`, constant line width, **dashed** near-zero bonds, **90th-percentile colour cut with the true
   max in the title**, `extend='max'` on the colourbar. Designed `k` is heavy-tailed (max/median 17
   healthy, 147 degenerate), so a `[min,max]` norm renders a dense panel as a flat dark mass in which
   correctly-dashed bonds read as a torn mesh — that is the B-4 figure defect, and it was the norm, not
   the data.
2. **ν(θ) and E(θ)** — Cartesian as the main panel, polar as companion. ν polar: radius `|ν|`, **blue
   where ν>0, red where ν<0**; E polar direct. Scalar ν and E are annotations on these panels, not a
   separate readout.

Square plot regions throughout. Conventions reimplemented in JS (the server sends data, not pictures);
`plotting.py` stays the authority for what they *are*.

**Deliberately NOT shown, and why:**

- **Per-triangle `C(s)` field maps — OMITTED as too revealing** (user's call). The residual head gives
  `G = (I+X)G_an(I+X)ᵀ` against the truth `C = (1+W)ᵀA(1+W)`, so `(1+W)ᵀ = Q(I+X)Q⁻¹` — **`W` is fully
  recoverable from the model's output.** Publishing the per-triangle field therefore exposes that the
  model predicts a per-triangle tensor and how it decomposes, which the bulk curves do not. See B.6/9.
- **The raw `C_eff` tensor** — same reasoning, and the curves carry what a reader needs.
- **Any deformation view — SKIPPED** (user's call). It could only be *macroscopic*: a faithful
  relaxed-node animation needs the non-affine displacement, i.e. `W`, and while `W` is recoverable it
  is exactly what is being withheld. And ν(θ)/E(θ) already covers all directions, so it adds nothing.
  *(Revisit only if the head is ever changed to predict `W` directly — `GNN_GUIDE.md` §11 lists that as
  untried.)*

**Controls:** library picker · family + parameter builder · `k` tuning (per-bond paint or pattern
presets) · **node drag**, guarded by a signed-area check so a drag cannot invert a triangle.

**Explanations:** every panel carries an optional "what am I looking at" disclosure, so one page serves
a collaborator, a general scientific reader, and the user's own exploration.

**B2 design view** *(after A0)*: ν(θ)/E(θ) with the **target dashed over the achieved curve**, plus a
**loss-against-iteration trace** — mirroring how the project already reports design results, so the
site stays consistent with its figures. Final state only is streamed; no per-iteration animation (that
would mean streaming intermediate state, raising cost and abuse surface on a free tier).

## B.4 Labelling — and no comparison on the site at all *(user, 2026-09-15)*

Every response panel is marked a **surrogate response**, predicted by a learned model. That is the
whole of it: no verification button, no accuracy dashboard, no defence of the numbers. It is a demo,
not a publication of results.

**`/compare` is DROPPED.** The user's constraint — the solver must not touch outside input — plus the
fact that the comparison is only ever for the user, leaves nothing for it to do on the site. The
comparison already exists locally and better: `Phase 5/m2/evaluate_v2.py` and
`Phase 5/verifications/m2_v3_report.py` score model, solver and sim in one pass on one split.
**Nothing physics-side is deployed.**

*(Recorded for the future: if a live comparison is ever wanted, the exact solver is the SAFER of the
two — `physical_homog` hard-crashes with an uncatchable native segfault on near-singular geometry,
which is a process death, not an exception. Neither is going online now.)*

## B.5 Phasing

| phase | contents | depends on |
|---|---|---|
| **B1 — forward** | library · build-your-own · k tuning · node drag · explanations | **nothing in Track A** |
| **B2 — gradient design** | "give me ν = −0.3" → descent on (k, positions) through the GNN, in-page | **A0.1** to exist at all, and **A0.2/A0.3** to say the gradient is usable — **both now satisfied** |

⚠ **B2 is the main cost vector.** An optimisation per request is seconds of CPU on a free tier, where a
forward pass is milliseconds. It needs a hard iteration cap, a wall-clock timeout, and a queue depth of
one per client.

⚠ **B2 must carry A0's trust region.** The gradient cosine collapses above `max|W|` ≈ 10
(`GRADIENT_FIDELITY.md`), and A0.3 found 2 of 20 targets where descent through the surrogate produced a
design **worse than doing nothing**. The design endpoint must refuse or flag requests that walk into
that regime, rather than returning a confident wrong answer.

## B.6 Security *(user: "all of the above and anything else you might think of")*

**Named by the user:**

1. **Unpublished research code stays private.** Deployable set is exactly: `model_v2.py`,
   `model_v3.py`, the inference half of `train_v3.prepare()`, **`Phase 2/mesh_build.py`** (verified to
   import only numpy + `scipy.spatial.Delaunay`), extracted point-cloud generators, and the checkpoint.
   **Excluded:** the solver, Phase 3, `verification_tools`, and `seeds.py` / `triangulation.py` — both
   import `_common`, which by its own comment "sets up all other sys.path and imports the solver stack".
2. **No link back to this repo or branch.** Fresh `git init`, no remote to MATERIALIZE, no submodule, no
   shared history. Also scrub the *attribution trail*: commit messages, and any `_commit()` provenance
   stamp copied along with a file.
3. **Abuse and cost control.** Rate limit per IP; hard caps on node count, triangle count and
   k-contrast; request timeout; CORS restricted to the user's own domain; B2's iteration cap above.
4. **The checkpoint.** **Ship a SANITISED checkpoint.** Inspected 2026-09-15: the file carries
   `holdout: bravais`, `w_max_cut: 10.0`, `huber: 1.0`, `epochs: 400`, `n_train: 25812`, `seed: 0`,
   `data: dataset_v2_s0.npz` — training protocol, dataset identity, filter threshold and sample count.
   Only `ns, nt, hidden, layers, use_star, use_global, head` + `state` are needed to rebuild the model;
   strip the rest.

**Added — things not named, that I would guard:**

5. **A crafted graph payload is a crash/DoS vector, not a formality.** The API accepts `tri_bond`,
   `bond_u/v` etc. and uses them as **array indices**; a malformed or hostile payload can go out of
   bounds or request an enormous allocation. Strict schema validation with explicit bounds checks on
   every index array, before anything reaches numpy or torch.
6. **Delaunay on user-supplied points** will hang on a large point set. Cap the count and time it.
7. **The preset library leaks the same way the checkpoint does** — family names, seeds, commit hashes,
   dataset ids. Sanitise the shipped JSON, not just the weights.
8. **Pin every dependency version** in the image; no credentials of any kind baked in; do not persist
   user input to logs.
9. **What the OUTPUT reveals is part of the attack surface, not just the code.** `W` is recoverable from
   a per-triangle prediction (`(1+W)ᵀ = Q(I+X)Q⁻¹`), so publishing per-triangle `C(s)` or the raw
   `C_eff` would leak the model's internal decomposition even with every file kept private. B.3 omits
   both; the API must not return them either, only the bulk curves the page draws.
10. **Model extraction is NOT preventable and should not be claimed as solved.** Anyone who can query
    the endpoint can distil an approximation of the surrogate; rate limits raise the cost and nothing
    more. For a public demo of a research surrogate that is probably acceptable — but it is the user's
    call, made knowingly, not a gap to paper over.

## B.7 The clone and how it deploys

**A Hugging Face Space IS a git repository.** That gives the cleanest answer to "no link to this repo":
there need be **no GitHub repository for the site at all**.

**Start with ONE Space serving both halves** — FastAPI serves the static frontend alongside the API. One
deploy, no CORS, no second account, fewest things to get wrong on a first pass. Suits "demo only, maybe
some fixes later".

```
  local private git repo   (source of truth; fresh init, no MATERIALIZE remote)
        |  git push  ->  the Space's own remote
  Hugging Face Space       (frontend + API; private at first, public when chosen)
```

**Split the frontend out only when the user's own domain matters** — a couple of hours, not a rewrite:

```
  static frontend  ->  Cloudflare Pages (preferred over github.io: deploys from a PRIVATE repo, and
                       gives free rate limiting / WAF in front) under the user's domain
        |  fetch(), CORS-restricted
  Hugging Face Space   (API only)
```

*Note:* **GitHub Pages on a free account requires a PUBLIC repo** (private-repo Pages needs Pro/Team).
The frontend holds nothing sensitive — no model, no physics, no research code, only the API URL — so
this is a minor point, recorded because the user asked for no loose ends.

### Setting up the Space — the actual steps

0. ⚠ **Verify the current free-tier limits** (RAM, sleep policy, card requirement). These change.
1. **Account** — `huggingface.co`, free, no card.
2. **New Space**: pick a name, **SDK = Docker**, **hardware = CPU basic (free)**, **visibility =
   Private** to begin with.
3. HF creates the Space **as a git repository** with remote
   `https://huggingface.co/spaces/<user>/<name>`.
4. **Access token**: Settings → Access Tokens → a **write** token. Used as the git password (or
   `huggingface-cli login`).
5. **Locally**: `git init` in the clone directory, add the files, `git remote add origin <space url>`,
   `git push`. **No GitHub involved at any point.**
6. **Files the Space needs at its root:**
   - `README.md` with HF front-matter (`title`, `emoji`, `sdk: docker`, `app_port: 7860`);
   - `Dockerfile` — `python:3.11-slim`, install requirements, copy the app, `CMD` runs uvicorn;
   - `requirements.txt` — **pinned**, and torch from the CPU index
     (`--index-url https://download.pytorch.org/whl/cpu`, else pip pulls multi-GB CUDA wheels);
   - the app, the **sanitised** checkpoint (3.2 MB — commit directly, no LFS needed), and the static
     frontend served by FastAPI.
7. **Push → HF builds the image**; build logs are visible in the Space UI.
8. **Going public** is one setting in Space Settings, whenever the user chooses.
9. **Redeploy** = commit and push again.

**Gotchas worth knowing in advance:** HF expects the app on **port 7860** unless `app_port` says
otherwise; the CPU torch index is not optional unless you want a multi-gigabyte image; and free Spaces
**sleep when idle**, which the static preset library hides.

**Separation is a property of the repo contents, not the hosting**, and is unaffected by any of the
above: the clone carries only the B.6 deployable set, has a fresh history, and has no remote, no
submodule and no attribution trail pointing at MATERIALIZE. One repo or two, both are clones.

## B — verification

| what | how |
|---|---|
| the deployed clone is clean | an import audit asserting the bundle imports no solver / Phase 3 / `verification_tools` / `seeds` / `triangulation`; a grep for MATERIALIZE paths and provenance stamps |
| the checkpoint is sanitised | the shipped file carries only `ns, nt, hidden, layers, use_star, use_global, head, state` — assert the provenance keys are absent |
| predictions match the research code | the deployed API and a local `T3.predict(net, T3.prepare(g))` agree to ≤1e-12 on a fixed set of meshes — the site must not silently diverge from the science |
| hostile input cannot crash or hang it | fuzz the API with malformed index arrays, out-of-range indices, oversized meshes and extreme `k`; every case returns an error, none kills the process |
| meshes are valid | `mesh_build.check_mesh_preconditions` plus a signed-area check on every input, including after a node drag |
| B2 is honest (when it ships) | achieved-vs-target scored against the **exact solver locally**, never on the host, before the feature is enabled; and the `max|W|` trust region enforced |

## B — cost

≈ 2–3 days for B1, no compute. B2 is ~1 further day.

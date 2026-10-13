# Downstream releases

Run **Release downstream vLLM stack** from Actions on `powerhrg`. The workflow
builds the Python router, operator and chart from one frozen source commit.
The chart is published as a Helm OCI artifact on GHCR, alongside the images.
Allow Actions to publish organization packages. Existing GHCR packages must
grant this repository Actions access. New GHCR packages are private by default;
change their visibility to public if unauthenticated image/OCI pulls are wanted.
These settings are not changed by the workflow.

```sh
gh workflow run powerhrg-release.yml --repo powerhome/vllm-production-stack \
  --ref powerhrg -f source_branch=powerhrg
```

The two optional inputs are `source_branch` (default `powerhrg`) and `output_tag`.
Source commits whose workflow files differ from `powerhrg` may require workflow
write permission for tagging/releasing, which `GITHUB_TOKEN` cannot grant. Use
matching workflow definitions when selecting another source branch.
With chart version `0.1.13`, an empty output tag selects
`vllm-stack-0.1.13-powerhrg-N`, where `N` is the highest existing matching suffix
plus one. Releases, drafts, remote tags, chart assets and package tags count.
For example, `...-9` followed by a gap selects `...-10`.

A manual tag must normalize to a Helm SemVer and Docker-compatible version.
`vllm-stack-0.1.13-powerhrg-7`, `v0.1.13-powerhrg-7` and
`0.1.13-powerhrg-7` all select chart version `0.1.13-powerhrg-7` and image tag
`v0.1.13-powerhrg-7`; the Git tag remains exactly as supplied. Existing tags,
releases or normalized package versions fail preflight. SemVer `+` metadata
cannot form these Docker tags and is rejected.

The chart defaults to
`ghcr.io/powerhome/vllm-production-stack/router:v<chart-version>`. The operator
is published as `ghcr.io/powerhome/vllm-production-stack/production-stack-operator`
with that same tag. Upstream vLLM defaults are preserved. Both images support
`linux/amd64`. The router image includes core routing modes only; it excludes
LMCache-backed `kvaware`/`loadaware` and experimental semantic caching, along with
their vLLM, PyTorch and CUDA dependencies. Upstream build recipes and dependency
definitions remain available for deployments requiring those optional features.
The workflow uses `self-hosted` runners. They need Linux x86-64, Python 3.11 or
newer, Docker with Buildx, gzip, and enough storage for the installed image plus
its exported archive. Build and
publication jobs report available archive storage and the Docker storage path.
The workflow does not remove preinstalled SDKs or alter runner configuration.
Publication uses private job-scoped Docker/Helm registry configurations under
`RUNNER_TEMP` and removes them even when publication fails.
Image jobs use a digest-pinned builder with at most two parallel build steps.
After artifact upload, an always-run cleanup removes only that run's loaded
image and dedicated builder/cache; it does not prune shared Docker caches.

## Install the chart

```sh
# Authenticate to GHCR first if the OCI package is private.
helm install stack oci://ghcr.io/powerhome/vllm-production-stack/charts/vllm-stack \
  --version 0.1.13-powerhrg-7 -f values.yaml
```

Always specify the chart version: `-powerhrg-N` is a SemVer prerelease suffix.
The release also attaches the packaged chart. The manifest records source SHA,
image/chart digests,
archive ordering and checksums; `SHA256SUMS` also covers the manifest.

## Load downloaded images

Download the release assets into one directory, then verify and load:

```sh
sha256sum --check SHA256SUMS
docker load --input operator-v0.1.13-powerhrg-7.tar.gz
# Router archives of 2 GiB or more are split into ordered 1 GiB parts.
cat router-v0.1.13-powerhrg-7.tar.gz.part[0-9][0-9][0-9] | docker load
# For an unsplit router archive, use docker load --input instead.
```

Publication spans GHCR and GitHub Releases. A failure after tag creation retains
its partial publication and fails the run without automatic cleanup.
Inspect the failed run and reserved draft before choosing
a new version. Publication failures need manual assessment rather than blindly
rerunning the publish job. The workflow pulls the published OCI chart and checks
its archive checksum before making the release public.

This serialized workflow exclusively owns the versioned router, operator and
chart tags in its GHCR namespace. It checks collisions during preparation and
again immediately before each push; a collision or API error aborts publication.
GHCR tags are mutable and these checks cannot prevent a concurrent external
writer from changing a tag. Other publishers must coordinate with this workflow;
use the recorded digests when consuming immutable artifacts.

Release dependencies are verified against
`scripts/powerhrg_chart_dependencies.json`, whose hashes are the OCI chart-content
layer digests. When updating chart dependencies, review and update those content
pins too. A source branch with unknown dependencies or missing lockfiles fails.

During future upstream updates, retain
`.github/workflows/powerhrg-release.yml` in the `update-project` manifest's
`downstream_workflows` allowlist. Keep inherited release workflows disabled.

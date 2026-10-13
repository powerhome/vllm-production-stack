#!/usr/bin/env python3
"""Bounded release operations for the manual downstream workflow (stdlib only)."""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "powerhome/vllm-production-stack"
FORMAT = "powerhrg-release/v1"
SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)
HELM_VERSION = "3.19.0"
HELM_SHA256 = "a7f81ce08007091b86d8bd696eb4d86b8d0f2e1b9f6c714be62f82f96a594496"
ASSET_LIMIT = 2 * 1024**3
PART_SIZE = 1024**3
BUFFER_SIZE = 1024 * 1024


def run(*args, cwd=None, timeout=300):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise RuntimeError("refusing an insecure download redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlsplit(newurl).hostname != "api.github.com":
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    def __init__(self):
        token = os.environ.get("GH_TOKEN")
        if not token:
            raise RuntimeError("GH_TOKEN is required")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": FORMAT,
        }
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, path, *, data=None, method=None, binary=False):
        headers = self.headers | ({"Accept": "application/octet-stream"} if binary else {})
        body = None if data is None else json.dumps(data).encode()
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request("https://api.github.com/" + path, body, headers, method=method)
        try:
            with self.opener.open(req, timeout=60) as response:
                payload = response.read(16 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"GitHub {path}: HTTP {exc.code}; no absence inferred") from exc
        if len(payload) > 16 * 1024 * 1024:
            raise RuntimeError("GitHub response exceeds size bound")
        return payload if binary else json.loads(payload)

    def pages(self, path):
        items = []
        for page in range(1, 101):
            separator = "&" if "?" in path else "?"
            batch = self.request(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(batch, list):
                raise RuntimeError("expected paginated GitHub array")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise RuntimeError("GitHub pagination exceeds 100 pages")

    def upload(self, release_id, path):
        # Stream archives rather than loading a GiB-sized asset into memory.
        connection = http.client.HTTPSConnection("uploads.github.com", timeout=120)
        endpoint = f"/repos/{REPOSITORY}/releases/{release_id}/assets?name=" + quote(path.name)
        try:
            connection.putrequest("POST", endpoint)
            for key, value in self.headers.items():
                connection.putheader(key, value)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(path.stat().st_size))
            connection.endheaders()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    connection.send(chunk)
            response = connection.getresponse()
            payload = response.read()
            if response.status != 201:
                raise RuntimeError(f"asset upload {path.name}: HTTP {response.status}")
            return json.loads(payload)
        finally:
            connection.close()


def quote(value):
    return urllib.parse.quote(value, safe="")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def normalized_version(tag):
    version = tag.removeprefix("vllm-stack-") if tag.startswith("vllm-stack-") else tag.removeprefix("v")
    match = SEMVER.fullmatch(version)
    if not match or any(part.isdigit() and len(part) > 1 and part.startswith("0")
                        for part in (match[4] or "").split(".")):
        raise RuntimeError(f"invalid Helm version derived from tag: {tag}")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", "v" + version):
        raise RuntimeError("version cannot form a Docker tag (including SemVer '+' metadata)")
    return version


def select_version(base, tags, releases, package_tags=(), override=""):
    # Asset names also count manual releases that used a different Git tag prefix.
    used = set(tags) | {release["tag_name"] for release in releases} | set(package_tags)
    for release in releases:
        used.update(asset["name"].removesuffix(".tgz") for asset in release["assets"]
                    if asset["name"].startswith("vllm-stack-") and asset["name"].endswith(".tgz"))
    if override:
        run("git", "check-ref-format", "refs/tags/" + override)
        if override in used:
            raise RuntimeError("requested tag/release already exists")
        version = normalized_version(override)
        for item in used:
            try:
                existing = normalized_version(item)
            except RuntimeError:
                continue
            if existing == version:
                raise RuntimeError("normalized chart/image version already exists")
        return override, version
    pattern = re.compile(re.escape(base) + r"-powerhrg-([1-9][0-9]*)")
    numbers = []
    for item in used:
        try:
            match = pattern.fullmatch(normalized_version(item))
        except RuntimeError:
            continue
        if match:
            numbers.append(int(match[1]))
    version = normalized_version(f"{base}-powerhrg-{max(numbers, default=0) + 1}")
    return "vllm-stack-" + version, version


def inventory(gh):
    releases = gh.pages(f"repos/{REPOSITORY}/releases")
    for release in releases:
        release["assets"] = gh.pages(f"repos/{REPOSITORY}/releases/{release['id']}/assets")
    tags = {line.split("\t", 1)[1].removeprefix("refs/tags/")
            for line in run("git", "ls-remote", "--tags", "--refs",
                            f"https://github.com/{REPOSITORY}.git").splitlines()}
    package_tags = set()
    names = {"vllm-production-stack/router", "vllm-production-stack/production-stack-operator",
             "vllm-production-stack/charts/vllm-stack"}
    # A successful authenticated listing is required even for first publication.
    for package in gh.pages("orgs/powerhome/packages?package_type=container"):
        if package["name"] in names:
            for version in gh.pages("orgs/powerhome/packages/container/" + quote(package["name"]) + "/versions"):
                package_tags.update(version["metadata"]["container"]["tags"])
    return tags, releases, package_tags


def scalar(text, name):
    matches = re.findall(r"^" + re.escape(name) + r":\s*([^\n#]+)", text, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one top-level {name}")
    return matches[0].strip().strip("\"'")


def prepare(args):
    gh = GitHub()
    repo = gh.request(f"repos/{REPOSITORY}")
    if repo["full_name"].lower() != REPOSITORY:
        raise RuntimeError("repository identity mismatch")
    run("git", "check-ref-format", "refs/heads/" + args.source_branch)
    source = gh.request(f"repos/{REPOSITORY}/git/ref/heads/{quote(args.source_branch)}")
    sha = source["object"]["sha"]
    content = gh.request(f"repos/{REPOSITORY}/contents/helm/Chart.yaml?ref={sha}")
    if content.get("encoding") != "base64":
        raise RuntimeError("unsupported chart content encoding")
    chart = base64.b64decode(content["content"], validate=False).decode()
    if scalar(chart, "name") != "vllm-stack":
        raise RuntimeError("source chart must be named vllm-stack")
    base = normalized_version(scalar(chart, "version"))
    tags, releases, package_tags = inventory(gh)
    tag, version = select_version(base, tags, releases, package_tags, args.output_tag)
    state = {
        "format": FORMAT, "repository": REPOSITORY, "source_branch": args.source_branch,
        "source_sha": sha, "source_version": base, "release_tag": tag, "chart_version": version,
        "image_tag": "v" + version,
        "run_id": os.environ["GITHUB_RUN_ID"],
    }
    write_json(args.state, state)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in state.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(state))


def install_helm(args):
    import tarfile

    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "helm.tar.gz"
    url = f"https://get.helm.sh/helm-v{HELM_VERSION}-linux-amd64.tar.gz"
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    if digest(archive) != HELM_SHA256:
        raise RuntimeError("Helm archive checksum mismatch")
    with tarfile.open(archive) as bundle, bundle.extractfile("linux-amd64/helm") as source:
        with (destination / "helm").open("wb") as output:
            shutil.copyfileobj(source, output)
    (destination / "helm").chmod(0o755)
    archive.unlink()


def stage_chart(args):
    state = read_json(args.state)
    source, target = Path(args.source), Path(args.destination)
    if not (source / "helm/Chart.lock").is_file():
        raise RuntimeError("source Chart.lock is required; dependency updates are forbidden")
    shutil.copytree(source / "helm", target, ignore=shutil.ignore_patterns("charts"))
    chart = (target / "Chart.yaml").read_text()
    chart, count = re.subn(r"^version:[^\n]*", "version: " + state["chart_version"], chart, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("source chart version is ambiguous")
    (target / "Chart.yaml").write_text(chart)
    original = (target / "values.yaml").read_text()
    match = re.search(r"^routerSpec:\s*\n((?:[ \t].*\n|#.*\n|\n)*)", original, re.MULTILINE)
    if not match:
        raise RuntimeError("source routerSpec structure is unsupported")
    body = match[1]
    for key, value in {"repository": f"ghcr.io/{REPOSITORY}/router", "tag": state["image_tag"]}.items():
        body, count = re.subn(r"^  " + key + r":[^\n]*", f"  {key}: {json.dumps(value)}", body, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError(f"expected one routerSpec.{key}")
    (target / "values.yaml").write_text(original[:match.start(1)] + body + original[match.end(1):])
    run("helm", "dependency", "build", str(target), "--skip-refresh")
    verify_dependencies(target)
    run("helm", "lint", str(target))
    rendered = run("helm", "template", "release-check", str(target))
    expected = f'image: "ghcr.io/{REPOSITORY}/router:{state["image_tag"]}"'
    if expected not in rendered:
        raise RuntimeError("rendered chart does not use the release router image")
    run("helm", "package", str(target), "--destination", args.packages)


def verify_dependencies(chart):
    pins = read_json(Path(__file__).with_name("powerhrg_chart_dependencies.json"))
    for dependency in (chart / "charts").glob("*"):
        if not dependency.is_file() or dependency.is_symlink() or dependency.name not in pins:
            raise RuntimeError(f"source dependency needs a reviewed content pin: {dependency.name}")
        if digest(dependency) != pins[dependency.name]:
            raise RuntimeError(f"chart dependency checksum mismatch: {dependency.name}")


def split_archive(args):
    path = Path(args.archive)
    parts = []
    if path.stat().st_size >= ASSET_LIMIT:
        with path.open("rb") as source:
            for number in range(1000):
                first = source.read(min(BUFFER_SIZE, PART_SIZE))
                if not first:
                    break
                part = path.with_name(f"{path.name}.part{number:03d}")
                with part.open("xb") as output:
                    output.write(first)
                    remaining = PART_SIZE - len(first)
                    while remaining:
                        chunk = source.read(min(BUFFER_SIZE, remaining))
                        if not chunk:
                            break
                        output.write(chunk)
                        remaining -= len(chunk)
                parts.append(part.name)
            else:
                raise RuntimeError("archive exceeds 1000 parts")
        path.unlink()
    else:
        parts = [path.name]
    write_json(path.parent / (args.component + ".json"), {
        "component": args.component, "parts": parts,
        "checksums": {name: digest(path.parent / name) for name in parts},
    })


def registry_digest(image):
    raw = run("docker", "buildx", "imagetools", "inspect", image, "--format", "{{json .Manifest}}")
    value = json.loads(raw)["digest"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise RuntimeError("invalid registry digest")
    return value


def publish(args):
    gh, state = GitHub(), read_json(args.state)
    directory = Path(args.packages)
    tags, releases, package_tags = inventory(gh)
    # Recheck all collisions immediately before the first publication side effect.
    select_version(state["source_version"], tags, releases, package_tags, state["release_tag"])
    manifest = state | {"images": {}, "chart": {}, "checksums": {}}
    for component in ("router", "operator"):
        metadata = read_json(directory / (component + ".json"))
        if metadata["component"] != component or not metadata["parts"]:
            raise RuntimeError("invalid archive metadata")
        for name in metadata["parts"]:
            if Path(name).name != name or digest(directory / name) != metadata["checksums"][name]:
                raise RuntimeError("archive checksum/name mismatch")
    chart = directory / f'vllm-stack-{state["chart_version"]}.tgz'
    if not chart.is_file():
        raise RuntimeError("chart package missing")
    # Atomic ref creation is the final protection against other tag writers.
    gh.request(f"repos/{REPOSITORY}/git/refs", data={
        "ref": "refs/tags/" + state["release_tag"], "sha": state["source_sha"],
    })
    release = gh.request(f"repos/{REPOSITORY}/releases", data={
        "tag_name": state["release_tag"], "target_commitish": state["source_sha"],
        "name": state["release_tag"], "draft": True,
    })
    print(f'Draft release {release["html_url"]} reserved; failures retain partial publication.', flush=True)
    for component, package in (("router", "router"), ("operator", "production-stack-operator")):
        metadata = read_json(directory / (component + ".json"))
        # docker load accepts gzip; stream the parts in their verified order.
        with subprocess.Popen(["docker", "load"], stdin=subprocess.PIPE) as loader:
            try:
                for name in metadata["parts"]:
                    with (directory / name).open("rb") as stream:
                        shutil.copyfileobj(stream, loader.stdin)
                loader.stdin.close()
                if loader.wait(timeout=900):
                    raise RuntimeError("docker load failed")
            except BaseException:
                loader.kill()
                loader.wait()
                raise
        image = f'ghcr.io/{REPOSITORY}/{package}:{state["image_tag"]}'
        run("docker", "push", image, timeout=1800)
        manifest["images"][component] = {"reference": image, "digest": registry_digest(image), "archives": metadata["parts"]}
        run("docker", "image", "rm", image)
    output = run("helm", "push", str(chart), f"oci://ghcr.io/{REPOSITORY}/charts")
    chart_ref = f'ghcr.io/{REPOSITORY}/charts/vllm-stack:{state["chart_version"]}'
    manifest["chart"] = {"reference": "oci://" + chart_ref.rsplit(":", 1)[0],
                          "digest": registry_digest(chart_ref), "archive": chart.name}
    verify_oci_chart(manifest["chart"]["reference"], state["chart_version"], chart, directory / "verified-chart")
    # Helm writes its progress to stderr; registry readback supplies the digest.
    print(output, end="")
    assets = [chart]
    for metadata in (read_json(directory / "router.json"), read_json(directory / "operator.json")):
        assets.extend(directory / name for name in metadata["parts"])
    manifest["checksums"] = {path.name: digest(path) for path in assets}
    write_json(directory / "release-manifest.json", manifest)
    assets.append(directory / "release-manifest.json")
    sums = directory / "SHA256SUMS"
    sums.write_text("".join(f"{digest(path)}  {path.name}\n" for path in assets))
    assets.append(sums)
    if len(assets) > 1000 or any(path.stat().st_size >= ASSET_LIMIT for path in assets):
        raise RuntimeError("release assets exceed GitHub limits")
    for path in assets:
        uploaded = gh.upload(release["id"], path)
        if uploaded["size"] != path.stat().st_size or uploaded.get("digest") != "sha256:" + digest(path):
            raise RuntimeError(f"release upload readback mismatch: {path.name}")
    ref = gh.request(f'repos/{REPOSITORY}/git/ref/tags/{quote(state["release_tag"])}')
    if ref["object"]["sha"] != state["source_sha"]:
        raise RuntimeError("release tag SHA changed")
    notes = (f'Source: `{state["source_branch"]}` at `{state["source_sha"]}`.\n\n'
             f'Chart version: `{state["chart_version"]}`.\n\n'
             + "\n".join(f'- `{image["reference"]}@{image["digest"]}`' for image in manifest["images"].values())
             + f'\nOCI chart: `{manifest["chart"]["reference"]}` (use `--version {state["chart_version"]}`).\n'
             + '\nCheck SHA256SUMS before loading archives. For split archives, concatenate .partNNN files in numeric order, then pass the stream to docker load.\n')
    gh.request(f'repos/{REPOSITORY}/releases/{release["id"]}', data={"draft": False, "body": notes, "make_latest": "false"}, method="PATCH")
    print("Release and GHCR OCI chart published and verified.")


def verify_oci_chart(reference, version, archive, destination):
    destination.mkdir()
    run("helm", "pull", reference, "--version", version, "--destination", str(destination))
    if digest(destination / archive.name) != digest(archive):
        raise RuntimeError("OCI chart checksum mismatch")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("prepare")
    cmd.add_argument("--source-branch", default="powerhrg")
    cmd.add_argument("--output-tag", default="")
    cmd.add_argument("--state", required=True)
    cmd.set_defaults(function=prepare)
    cmd = commands.add_parser("install-helm")
    cmd.add_argument("--destination", required=True)
    cmd.set_defaults(function=install_helm)
    cmd = commands.add_parser("stage-chart")
    for name in ("source", "state", "destination", "packages"):
        cmd.add_argument("--" + name, required=True)
    cmd.set_defaults(function=stage_chart)
    cmd = commands.add_parser("split-archive")
    cmd.add_argument("--archive", required=True)
    cmd.add_argument("--component", choices=("router", "operator"), required=True)
    cmd.set_defaults(function=split_archive)
    cmd = commands.add_parser("publish")
    for name in ("state", "packages"):
        cmd.add_argument("--" + name, required=True)
    cmd.set_defaults(function=publish)
    args = parser.parse_args()
    try:
        args.function(args)
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Release operation failed: {exc}\nNo published resources were deleted or overwritten.\n")


if __name__ == "__main__":
    main()

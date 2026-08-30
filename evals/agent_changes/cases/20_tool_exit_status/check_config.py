import subprocess


def validate(path: str) -> bool:
    result = subprocess.run(["config-check", path], check=False)
    return result.returncode != 127

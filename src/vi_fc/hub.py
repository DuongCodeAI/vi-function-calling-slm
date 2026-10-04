"""Đẩy kết quả lên HF Hub (tuỳ chọn). Repo nằm dưới tài khoản của HF_TOKEN, hoặc HF_USER nếu đặt (vd đẩy vào org)."""

import os


def hf_user(token: str | None = None) -> str:
    if os.environ.get("HF_USER"):
        return os.environ["HF_USER"]
    from huggingface_hub import HfApi

    return HfApi(token=token).whoami()["name"]


def repo_id(name: str, token: str | None = None) -> str:
    """Tên repo không kèm user (vi-fc-qwen3-1.7b) -> <user>/vi-fc-qwen3-1.7b; đã có "/" thì giữ nguyên."""
    return name if "/" in name else f"{hf_user(token)}/{name}"


def push(path: str, name: str, path_in_repo: str | None = None, repo_type: str = "model",
         token: str | None = None, **kw) -> str | None:
    """Thiếu token hoặc push lỗi (token chỉ có quyền đọc, mạng...) thì in ra rồi bỏ qua, không làm dừng notebook."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("không có HF_TOKEN, bỏ qua push", name)
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo = repo_id(name, token)
        api.create_repo(repo, repo_type=repo_type, exist_ok=True)
        if os.path.isdir(path):
            api.upload_folder(folder_path=path, repo_id=repo, repo_type=repo_type, path_in_repo=path_in_repo, **kw)
        else:
            api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo or os.path.basename(path), repo_id=repo,
                            repo_type=repo_type)
    except Exception as e:
        print(f"push {name} lỗi, bỏ qua: {e}")
        return None
    print("đã push", repo)
    return repo

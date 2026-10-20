from huggingface_hub import HfApi, ModelCard


def upload_to_huggingface(
    local_path: str,
    repo_id: str,
    path_in_repo: str = "",
    repo_type: str = "model",
    token: str | None = None,
    model_card: ModelCard | str | None = None,
):
    """Upload a folder to a HuggingFace repository.

    Args:
        local_path (str): Local directory containing files to upload.
        repo_id (str): HuggingFace repository ID.
        path_in_repo (str): Destination folder path inside the repository. Defaults to "".
        repo_type (str): Repository type ('model', 'dataset', or 'space').
        token (str): HuggingFace API token. Falls back to the cached login token if None.
        model_card (ModelCard | str, optional): Model card uploaded as ``README.md`` to the repository root,
            independent of ``path_in_repo``, e.g. from :func:`~spherinator.models.create_model_card`.
            Defaults to None.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    )
    print(f"Uploaded {local_path} to {repo_id}/{path_in_repo}")

    if model_card is not None:
        api.upload_file(
            path_or_fileobj=str(model_card).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload model card",
        )
        print(f"Uploaded model card to {repo_id}/README.md")

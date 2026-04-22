"""
Grid plot for steered images in a directory (e.g. apply_steering_with_injection_flux_*.py output).

Filenames look like:
  00_prompt_..._s_50.0_simg_0.0_mask_False_v_diff.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from PIL import Image


PathLike = Union[str, Path]


def _stem_sort_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    m = re.match(r"^(\d+)_", stem)
    if m:
        return (0, int(m.group(1)), stem)
    return (1, 0, stem)


def list_images_by_strength(
    directory: PathLike,
    strength: Optional[float] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> List[Path]:
    """
    All images in ``directory`` (non-recursive), optionally filtered by
    ``_s_{strength}_`` in the basename (matches how steering scripts name files).
    Sorted by leading numeric index ``NN_`` if present, else lexicographically.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ext_set]

    if strength is not None:
        # e.g. _s_50.0_ — avoid accidental match on simg_ by requiring _s_<num>_
        token = f"_s_{float(strength)}_"
        files = [p for p in files if token in p.name]

    files.sort(key=_stem_sort_key)
    return files


def plot_steering_grid(
    directory: PathLike,
    indices: Sequence[int],
    strength: Optional[float] = None,
    *,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    ncols: int = 4,
    figsize_per: Tuple[float, float] = (3.5, 3.5),
    titles: Optional[Sequence[Optional[str]]] = None,
    suptitle: Optional[str] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> None:
    """
    Plot a subset of images from ``directory`` by **list index** into the
    sorted list returned by :func:`list_images_by_strength`.

    Parameters
    ----------
    directory
        Folder with images (e.g. ``.../steered``).
    indices
        0-based indices into the sorted file list (e.g. ``[0, 3, 7, 19]``).
    strength
        If set, only files whose names contain ``_s_{strength}_`` are considered.
    save_path
        If set, save figure to this path.
    ncols
        Number of columns in the grid.
    figsize_per
        (width, height) in inches per subplot cell (total figure scales with grid).
    titles
        Optional titles per plotted panel (same length as ``indices``); use ``None`` for a slot to show default (#idx).
    suptitle
        Optional figure title.
    """
    paths = list_images_by_strength(directory, strength=strength, extensions=extensions)
    if not paths:
        raise FileNotFoundError(
            f"No images in {directory!r}"
            + (f" matching strength s_{strength}" if strength is not None else "")
        )

    n = len(indices)
    if n == 0:
        return

    bad = [i for i in indices if i < 0 or i >= len(paths)]
    if bad:
        raise IndexError(
            f"indices {bad} out of range for {len(paths)} files "
            f"(strength={strength!r})"
        )

    nrows = (n + ncols - 1) // ncols
    fig_w = figsize_per[0] * ncols
    fig_h = figsize_per[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    if titles is not None and len(titles) != n:
        raise ValueError(f"titles length {len(titles)} != len(indices) {n}")

    for slot, idx in enumerate(indices):
        r, c = divmod(slot, ncols)
        ax = axes[r][c]
        path = paths[idx]
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        if titles is not None and titles[slot] is not None:
            ax.set_title(titles[slot], fontsize=9)
        else:
            ax.set_title(f"#{idx}\n{path.name[:40]}…" if len(path.name) > 42 else f"#{idx}\n{path.name}", fontsize=8)

    for slot in range(n, nrows * ncols):
        r, c = divmod(slot, ncols)
        axes[r][c].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    elif strength is not None:
        fig.suptitle(f"{Path(directory).name} — s_{strength}", fontsize=11)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_indices_for_prompt(
    directory: PathLike,
    prompt_indices: Iterable[int],
    strength: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Same as :func:`plot_steering_grid`, but ``prompt_indices`` are the leading
    ``NN`` in filenames (e.g. ``5`` → file starting with ``05_`` or ``5_``).
    """
    paths = list_images_by_strength(directory, strength=strength)
    want = set(int(x) for x in prompt_indices)
    list_idx: List[int] = []
    for i, p in enumerate(paths):
        m = re.match(r"^(\d+)_", p.stem)
        if m and int(m.group(1)) in want:
            list_idx.append(i)
    if len(list_idx) != len(want):
        found = {int(re.match(r"^(\d+)_", p.stem).group(1)) for p in paths if re.match(r"^(\d+)_", p.stem)}
        missing = sorted(want - found)
        if missing:
            raise FileNotFoundError(f"No file for prompt indices {missing} under {directory!r}")

    plot_steering_grid(directory, list_idx, strength=strength, **kwargs)

from pathlib import Path

import pytest

from cellphi.pipeline import find_images


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "data"
    touch(root / "a.jpg")
    touch(root / "day1" / "b.TIF")
    touch(root / "day1" / "wellA" / "c.png")
    touch(root / "day2" / "a.jpg")  # same file name as the top-level one
    touch(root / "notes.txt")
    touch(root / ".hidden" / "x.jpg")
    touch(root / "results" / "a_annotated.jpg")
    touch(root / "day1" / "old_masks.png")
    return root


def names(items):
    return [i.name for i in items]


def test_recursive_search(tree):
    found = names(find_images([tree], exclude_dirs=[tree / "results"]))
    assert found == ["a.jpg", "day1/b.TIF", "day1/wellA/c.png", "day2/a.jpg"]


def test_non_recursive(tree):
    assert names(find_images([tree], recursive=False)) == ["a.jpg"]


def test_skips_output_folder_and_cellphi_outputs_even_if_not_excluded(tree):
    found = names(find_images([tree]))
    assert not any("annotated" in n or "masks" in n for n in found)


def test_several_input_folders_keep_their_names(tmp_path):
    touch(tmp_path / "Cortical" / "Plate 1" / "A1" / "x.jpg")
    touch(tmp_path / "Midbrain" / "Plate 1" / "A1" / "y.jpg")
    found = names(find_images([tmp_path / "Cortical", tmp_path / "Midbrain"]))
    assert found == ["Cortical/Plate 1/A1/x.jpg", "Midbrain/Plate 1/A1/y.jpg"]


def test_unique_names_for_same_file_name(tmp_path):
    touch(tmp_path / "a" / "img.jpg")
    touch(tmp_path / "b" / "img.jpg")
    found = names(find_images([tmp_path / "a" / "img.jpg", tmp_path / "b" / "img.jpg"]))
    assert found == ["img.jpg", "b/img.jpg"]


def test_files_and_duplicates(tree):
    found = find_images([tree / "a.jpg", tree], exclude_dirs=[tree / "results"])
    assert names(found)[0] == "a.jpg"
    assert len(found) == 4  # a.jpg not counted twice


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_images([tmp_path / "nope"])

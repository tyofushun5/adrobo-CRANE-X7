import sys
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from train.dreamer_train import Config, evaluation, make_env, main, train

__all__ = ["Config", "evaluation", "make_env", "main", "train"]

if __name__ == "__main__":
    main()

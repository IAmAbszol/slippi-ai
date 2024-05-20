import enum
import numpy as np
import os
import shutil
import pickle
import multiprocessing
import zlib

from absl import app
from absl import flags

import melee
from slippi_ai import (embed, utils)

class GameStateWrapper:
  def __init__(self, game_state):
    self.game_state = game_state

  def add_player_attribute(self, port, value):
    setattr(self, f'p{port}', value)

  def __getattr__(self, name):
    if name in self.__dict__:
      return self.__dict__[name]
    return getattr(self.game_state, name)

def modify_in_place(obj, depth=0, max_depth=20):
  if depth > max_depth:
    print(f"Reached max recursion depth at depth {depth}.")
    return

  if isinstance(obj, (np.ndarray, np.number, np.bool_)):
    print("Encountered a NumPy array or number; skipping.")
    return

  for key in dir(obj):
    if key.startswith('_') or key in ['count', 'index']:
      continue

    value = getattr(obj, key)

    if isinstance(value, enum.Enum):
      setattr(obj, key, value.value)
    elif isinstance(value, dict):
      for dict_key, dict_value in value.items():
        if isinstance(dict_value, enum.Enum):
          value[dict_key] = dict_value.value
        if isinstance(dict_key, enum.Enum):
          dict_value = value.pop(dict_key)
          value[dict_key.name] = dict_value
    elif isinstance(value, (list, tuple)):
      for i, element in enumerate(value):
        modify_in_place(element, depth=depth + 1)
    elif not isinstance(value, (int, float, str, bool)):
      modify_in_place(value, depth=depth + 1)


FLAGS = flags.FLAGS
flags.DEFINE_integer('cores', 1, 'number of cores')
flags.DEFINE_boolean('compress', True, 'Compress with zlib.')

flags.DEFINE_string('src_dir', 'training_data', 'Folder with slippi replays.')
flags.DEFINE_string('dst_dir', 'training_data/compressed_data', 'Where to create the dataset.')

def read_gamestates(replay_path):
  print("Reading from ", replay_path)
  console = melee.Console(is_dolphin=False,
                          allow_old_version=True,
                          path=replay_path)
  console.connect()

  gamestate = console.step()
  port_map = dict(zip(gamestate.player.keys(), [1, 2]))

  def fix_state(s):
    s.player = {port_map[p]: v for p, v in s.player.items()}
    for port, v in enumerate(s.player.values()):
      s.add_player_attribute(port, v)

  while gamestate:
    modify_in_place(gamestate)
    gamestate_wrapper = GameStateWrapper(gamestate)
    fix_state(gamestate_wrapper)
    yield gamestate_wrapper
    gamestate = console.step()

# TODO: enable speeds?
embed_game = embed.make_game_embedding()

def game_to_numpy(replay_path):
  states = read_gamestates(replay_path)
  states = map(embed_game.from_state, states)
  return utils.batch_nest(states)

def slp_to_pkl(src_dir, dst_dir, name, compress=False):
  src = os.path.join(src_dir, name)
  assert os.path.isfile(src)
  dst = os.path.join(dst_dir, name + '.pkl')
  if os.path.isfile(dst): return
  obj = game_to_numpy(src)
  obj_bytes = pickle.dumps(obj)
  if compress:
    obj_bytes = zlib.compress(obj_bytes)
  with open(dst, 'wb') as f:
    f.write(obj_bytes)

def batch_slp_to_pkl(src_dir, dst_dir, names, compress=False, cores=1):
  os.makedirs(dst_dir, exist_ok=True)

  dst_files = set(os.listdir(dst_dir))
  def is_new(name):
    return (name + '.pkl') not in dst_files
  names = list(filter(is_new, names))
  print(f"Converting {len(names)} replays.")

  # to see error messages
  if cores == 1:
    for name in names:
      # try:
      slp_to_pkl(src_dir, dst_dir, name, compress)
      # except Exception:
      #   print('Bad replay file', name)
    return

  with multiprocessing.Pool(cores) as pool:
    results = []
    for name in names:
      results.append(pool.apply_async(
          slp_to_pkl, [src_dir, dst_dir, name, compress]))
    for r in results:
      r.wait()

def main(_):
  subset = os.listdir(FLAGS.src_dir)
  subset = set(subset)

  batch_slp_to_pkl(
      FLAGS.src_dir,
      FLAGS.dst_dir,
      subset,
      FLAGS.compress,
      FLAGS.cores)

if __name__ == '__main__':
  app.run(main)

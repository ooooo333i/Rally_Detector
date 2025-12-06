# after deleting useless sequences, renumber sequences and frames

import os
import shutil

def renumber_sequences(base_dir):
    
    seqs = [d for d in os.listdir(base_dir) if d.startswith("seq_")]
    seqs.sort()

    print(f"[INFO] Found {len(seqs)} sequences.")

    new_index = 1
    for old_name in seqs:
        old_path = os.path.join(base_dir, old_name)
        new_name = f"seq_{new_index:05d}"
        new_path = os.path.join(base_dir, new_name)

        if old_name != new_name:
            os.rename(old_path, new_path)
            print(f"[RENAME] {old_name} → {new_name}")
        else:
            print(f"[OK] {old_name} (already correct)")

        frames = [f for f in os.listdir(new_path) if f.lower().endswith(".jpg")]
        frames.sort()

        tmp_dir = os.path.join(new_path, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        for i, frame in enumerate(frames, start=1):
            old_frame_path = os.path.join(new_path, frame)
            new_frame_name = f"{i:05d}.jpg"
            new_frame_path = os.path.join(tmp_dir, new_frame_name)
            shutil.move(old_frame_path, new_frame_path)

        for f in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, f), new_path)

        os.rmdir(tmp_dir)

        print(f"    → Frames renumbered ({len(frames)} frames)")

        new_index += 1

    print("[DONE] All sequences and frames renumbered successfully!")

renumber_sequences("Data/train/sequences")
renumber_sequences("Data/validation/sequences")
renumber_sequences("Data/test/sequences")
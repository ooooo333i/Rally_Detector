import os
import shutil

def renumber_sequences(base_dir):
    

    # seq 폴더만 추출
    seqs = [d for d in os.listdir(base_dir) if d.startswith("seq_")]
    seqs.sort()  # 이름순 정렬

    print(f"[INFO] Found {len(seqs)} sequences.")

    new_index = 1
    for old_name in seqs:
        old_path = os.path.join(base_dir, old_name)
        new_name = f"seq_{new_index:05d}"
        new_path = os.path.join(base_dir, new_name)

        # 폴더명 변경
        if old_name != new_name:
            os.rename(old_path, new_path)
            print(f"[RENAME] {old_name} → {new_name}")
        else:
            print(f"[OK] {old_name} (already correct)")

        # 내부 프레임 정렬
        frames = [f for f in os.listdir(new_path) if f.lower().endswith(".jpg")]
        frames.sort()

        # temp 폴더 생성해서 프레임 이름 충돌 방지
        tmp_dir = os.path.join(new_path, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # 모든 프레임을 temp로 이동 + 재이름 지정
        for i, frame in enumerate(frames, start=1):
            old_frame_path = os.path.join(new_path, frame)
            new_frame_name = f"{i:05d}.jpg"
            new_frame_path = os.path.join(tmp_dir, new_frame_name)
            shutil.move(old_frame_path, new_frame_path)

        # temp 안의 파일을 원래 위치로 이동
        for f in os.listdir(tmp_dir):
            shutil.move(os.path.join(tmp_dir, f), new_path)

        # temp 폴더 삭제
        os.rmdir(tmp_dir)

        print(f"    → Frames renumbered ({len(frames)} frames)")

        new_index += 1

    print("[DONE] All sequences and frames renumbered successfully!")


renumber_sequences("Data/sequences") 
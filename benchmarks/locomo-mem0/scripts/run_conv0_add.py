"""Ingest only conv0 (Caroline <-> Melanie) into Mem0 platform."""

import json
import os
import sys
import threading
import time

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from mem0 import MemoryClient

custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


def main():
    client = MemoryClient(
        api_key=os.getenv("MEM0_API_KEY"),
        org_id=os.getenv("MEM0_ORGANIZATION_ID"),
        project_id=os.getenv("MEM0_PROJECT_ID"),
    )

    client.update_project(custom_instructions=custom_instructions)

    with open("dataset/locomo10.json", "r") as f:
        data = json.load(f)

    item = data[0]
    idx = 0
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    speaker_a_id = f"{speaker_a}_{idx}"
    speaker_b_id = f"{speaker_b}_{idx}"

    print(f"Conv 0: {speaker_a} <-> {speaker_b}")
    print(f"User IDs: {speaker_a_id}, {speaker_b_id}")

    print(f"\nDeleting existing memories for {speaker_a_id}...")
    client.delete_all(user_id=speaker_a_id)
    print(f"Deleting existing memories for {speaker_b_id}...")
    client.delete_all(user_id=speaker_b_id)
    print("Existing memories cleared.\n")

    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.replace("session_", ""))
    )

    print(f"Total sessions: {len(session_keys)}")
    total_start = time.time()

    batch_size = 2
    errors = []

    for session_key in session_keys:
        date_time_key = session_key + "_date_time"
        timestamp = conversation.get(date_time_key, "")
        chats = conversation[session_key]

        print(f"\n{'='*50}")
        print(f"[{session_key}] {timestamp} — {len(chats)} messages")
        print(f"{'='*50}")

        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

        def add_for_speaker(speaker_id, msgs, label):
            for i in range(0, len(msgs), batch_size):
                batch = msgs[i:i + batch_size]
                for attempt in range(3):
                    try:
                        client.add(batch, user_id=speaker_id, version="v2", metadata={"timestamp": timestamp})
                        break
                    except Exception as e:
                        if attempt < 2:
                            print(f"  [{label}] Retry {attempt+1}: {e}")
                            time.sleep(2)
                        else:
                            err_msg = f"[{label}] {session_key} batch {i}: {e}"
                            print(f"  FAILED: {err_msg}")
                            errors.append(err_msg)
                print(f"  [{label}] batch {i//batch_size + 1}/{(len(msgs) + batch_size - 1)//batch_size} done")

        t1 = time.time()

        thread_a = threading.Thread(target=add_for_speaker, args=(speaker_a_id, messages, speaker_a_id))
        thread_b = threading.Thread(target=add_for_speaker, args=(speaker_b_id, messages_reverse, speaker_b_id))

        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

        elapsed = time.time() - t1
        print(f"  {session_key} done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print(f"\n{'='*50}")
    print(f"Conv 0 ingestion complete!")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

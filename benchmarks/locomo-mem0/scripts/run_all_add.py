"""Ingest conv1-conv9 into Mem0 platform (conv0 already done)."""

import json
import os
import sys
import threading
import time

from dotenv import load_dotenv

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


def ingest_conversation(client, data, idx):
    item = data[idx]
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    speaker_a_id = f"{speaker_a}_{idx}"
    speaker_b_id = f"{speaker_b}_{idx}"

    session_keys = sorted(
        [k for k in conversation.keys() if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.replace("session_", ""))
    )
    total_msgs = sum(len(conversation[s]) for s in session_keys)

    print(f"\n{'#'*60}")
    print(f"# CONV {idx}: {speaker_a} <-> {speaker_b}")
    print(f"# Sessions: {len(session_keys)} | Messages: {total_msgs}")
    print(f"{'#'*60}")

    print(f"  Deleting existing memories for {speaker_a_id}...")
    client.delete_all(user_id=speaker_a_id)
    print(f"  Deleting existing memories for {speaker_b_id}...")
    client.delete_all(user_id=speaker_b_id)

    batch_size = 2
    errors = []
    conv_start = time.time()

    for session_key in session_keys:
        date_time_key = session_key + "_date_time"
        timestamp = conversation.get(date_time_key, "")
        chats = conversation[session_key]

        print(f"  [{session_key}] {timestamp} — {len(chats)} msgs", end="", flush=True)

        messages = []
        messages_reverse = []
        for chat in chats:
            if chat["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
            elif chat["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})

        def add_for_speaker(speaker_id, msgs, err_list):
            for i in range(0, len(msgs), batch_size):
                batch = msgs[i:i + batch_size]
                for attempt in range(3):
                    try:
                        client.add(batch, user_id=speaker_id, version="v2", metadata={"timestamp": timestamp})
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            err_list.append(f"{speaker_id}/{session_key}/batch{i}: {e}")

        t1 = time.time()
        thread_a = threading.Thread(target=add_for_speaker, args=(speaker_a_id, messages, errors))
        thread_b = threading.Thread(target=add_for_speaker, args=(speaker_b_id, messages_reverse, errors))
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
        print(f" -> {time.time()-t1:.1f}s")

    elapsed = time.time() - conv_start
    print(f"  Conv {idx} done: {int(elapsed//60)}m {int(elapsed%60)}s, errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"    ERR: {e}")
    return len(errors)


def main():
    client = MemoryClient(
        api_key=os.getenv("MEM0_API_KEY"),
        org_id=os.getenv("MEM0_ORGANIZATION_ID"),
        project_id=os.getenv("MEM0_PROJECT_ID"),
    )
    client.update_project(custom_instructions=custom_instructions)

    with open("dataset/locomo10.json") as f:
        data = json.load(f)

    start_conv = 1
    end_conv = 10

    print("=" * 60)
    print(f"LOCOMO INGEST: conv{start_conv} - conv{end_conv-1}")
    print("=" * 60)

    total_start = time.time()
    total_errors = 0
    status = {}

    for idx in range(start_conv, end_conv):
        err_count = ingest_conversation(client, data, idx)
        total_errors += err_count
        status[idx] = "OK" if err_count == 0 else f"{err_count} errors"

        with open("results/ingest_status.json", "w") as f:
            json.dump({"completed": list(range(start_conv, idx+1)), "status": status,
                        "total_errors": total_errors}, f, indent=2)

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    mins = int((total_elapsed % 3600) // 60)
    secs = int(total_elapsed % 60)

    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)
    print(f"Conversations: {start_conv}-{end_conv-1}")
    print(f"Total time: {hours}h {mins}m {secs}s")
    print(f"Total errors: {total_errors}")
    for idx, s in status.items():
        print(f"  Conv {idx}: {s}")
    print("=" * 60)


if __name__ == "__main__":
    main()

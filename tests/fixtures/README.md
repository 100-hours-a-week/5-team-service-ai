# Fixture schemas

## users.jsonl
Fields per line:
- user_id: integer user identifier
- reading_volume_id: integer bucket for reading volume
- purpose_ids: list of integers (reading purposes)
- genre_ids: list of integers (preferred genres)

## meetings.jsonl
Fields per line:
- id: meeting identifier
- reading_genre_id: genre bucket for the meeting
- title: short name
- description: free-text summary
- status: RECRUITING | FINISHED | CANCELED
- capacity: max participants
- current_count: current participants (<= capacity)
- leader_intro: short intro for host/leader

## logs.jsonl
Fields per line:
- user_id: who saw/interacted
- meeting_id: meeting reference
- event_type: impression | click | join
- dwell_sec: optional int; present mostly for click/join events
- ts: ISO timestamp string

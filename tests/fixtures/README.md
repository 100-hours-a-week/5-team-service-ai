# Fixture schemas

## users.jsonl
Fields per line:
- user_id: integer user identifier
- reading_volume_code: string bucket for reading volume (e.g., ONE_OR_LESS, TWO_TO_THREE, ...)
- purpose_codes: list of strings (reading purposes)
- genre_codes: list of strings (preferred genres)

## meetings.jsonl
Fields per line:
- id: meeting identifier
- reading_genre_code: genre bucket for the meeting
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

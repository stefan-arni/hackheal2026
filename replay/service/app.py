"""FastAPI service: ingest frames, call fal, post-process, serve bundles. (Tier 1)

Routes (see REPLAY_SPEC.md "Interfaces"):
    POST /replay/{trialId}/frame   multipart: jpeg, mask?, t, crop, frame_size, kind, gravity?
    POST /replay/{trialId}/end     JSON: events, patient_height_cm, landmarks_2d?
    GET  /replay/{trialId}/meta.json
    GET  /replay/{trialId}/verts.bin
"""

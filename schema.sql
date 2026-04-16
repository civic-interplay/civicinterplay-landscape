-- Civic Interplay — Supabase schema
-- Run this in the Supabase SQL editor

CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Briefs ─────────────────────────────────────────────────────────────────
CREATE TABLE briefs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz DEFAULT now(),
  participant_name text,
  practice_area text,
  what_im_working_on text,
  question_im_carrying text,
  what_push_i_want text,
  embedding vector(768),
  x float,
  y float,
  density float
);

-- ─── Tags ───────────────────────────────────────────────────────────────────
CREATE TABLE tags (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  label text UNIQUE NOT NULL,
  is_seed boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- ─── Brief ↔ Tag join ──────────────────────────────────────────────────────
CREATE TABLE brief_tags (
  brief_id uuid REFERENCES briefs(id) ON DELETE CASCADE,
  tag_id uuid REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY (brief_id, tag_id)
);

-- ─── Edges (declared connections between briefs) ───────────────────────────
CREATE TABLE edges (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  from_brief_id uuid REFERENCES briefs(id) ON DELETE CASCADE,
  to_brief_id uuid REFERENCES briefs(id) ON DELETE CASCADE,
  edge_type text, -- 'resonance' | 'tension' | 'question' | 'facilitator'
  created_at timestamptz DEFAULT now()
);

-- ─── Index for cosine similarity search ────────────────────────────────────
CREATE INDEX ON briefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- ─── Seed tags ─────────────────────────────────────────────────────────────
INSERT INTO tags (label, is_seed) VALUES
  ('water', true),
  ('country', true),
  ('infrastructure', true),
  ('ecology', true),
  ('data sovereignty', true),
  ('collective intelligence', true),
  ('urban systems', true),
  ('storytelling', true),
  ('governance', true),
  ('more-than-human', true),
  ('community practice', true),
  ('technology critique', true);

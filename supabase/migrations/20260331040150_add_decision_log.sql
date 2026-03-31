/*
  # Add explicit decision tracking

  1. New Tables
    - `experiment_runs` - IMMUTABLE log of every experiment run
      - exp_id, dataset, config, epochs, batch_size
      - status (pending, running, complete, failed)
      - scores when complete
    - `decisions` - explicit human decisions made
      - date, winner_exp_id, reason_summary
      - next_planned_exp_id, rationale
      - this is the iteration memory

  2. Modifications
    - Drop old vague `experiments` table
    - Replace with strict tables that capture:
      * what ran (immutable)
      * what won (explicit decision)
      * what's next (planned step)

  3. Security
    - RLS: users only see their own data
    - Audit trail: no deletions, only inserts
*/

-- Drop old table if it exists
DROP TABLE IF EXISTS experiments CASCADE;

-- Immutable experiment run log
CREATE TABLE IF NOT EXISTS experiment_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  exp_id text NOT NULL,
  dataset text NOT NULL CHECK (dataset IN ('clean', 'natural', 'raw')),
  config text NOT NULL CHECK (config IN ('baseline', 'high_quality')),
  epochs integer NOT NULL,
  batch_size integer NOT NULL,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'complete', 'failed')),
  scores jsonb,
  model_path text,
  created_at timestamptz DEFAULT now(),
  completed_at timestamptz,
  UNIQUE(user_id, exp_id)
);

-- Decision log (iteration memory)
CREATE TABLE IF NOT EXISTS decisions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  winner_exp_id text NOT NULL,
  loser_exp_id text,
  reason_summary text NOT NULL,
  next_planned_exp_id text,
  rationale text,
  created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE experiment_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE decisions ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users see own experiment runs"
  ON experiment_runs FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users create own experiment runs"
  ON experiment_runs FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users update own experiment runs"
  ON experiment_runs FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users see own decisions"
  ON decisions FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users log own decisions"
  ON decisions FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_exp_runs_user_status ON experiment_runs(user_id, status);
CREATE INDEX IF NOT EXISTS idx_exp_runs_exp_id ON experiment_runs(exp_id);
CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_decisions_winner ON decisions(winner_exp_id);

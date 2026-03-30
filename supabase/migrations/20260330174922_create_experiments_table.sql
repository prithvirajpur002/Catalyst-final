/*
  # Create experiments tracking table

  1. New Tables
    - `experiments`
      - `id` (uuid, primary key) - unique experiment identifier
      - `exp_id` (text, unique) - experiment name (exp_001, exp_002, etc)
      - `dataset` (text) - which dataset was used (clean, natural)
      - `config` (text) - config name (baseline, high_quality)
      - `epochs` (integer) - training epochs
      - `batch_size` (integer) - batch size
      - `mode` (text) - preprocessing mode
      - `status` (text) - 'running', 'complete', 'failed'
      - `scores` (jsonb) - {naturalness, clarity, identity, composite}
      - `model_path` (text) - path to trained model
      - `index_path` (text) - path to FAISS index
      - `created_at` (timestamp)
      - `completed_at` (timestamp)
      - `user_id` (uuid, foreign key to auth.users)

  2. Security
    - Enable RLS on experiments table
    - Users can only view/manage their own experiments

  3. Indexes
    - (user_id, exp_id) for fast lookups
    - (status) for finding running/failed experiments
*/

CREATE TABLE IF NOT EXISTS experiments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  exp_id text NOT NULL UNIQUE,
  dataset text NOT NULL,
  config text NOT NULL,
  epochs integer NOT NULL,
  batch_size integer NOT NULL,
  mode text NOT NULL,
  status text DEFAULT 'pending',
  scores jsonb,
  model_path text,
  index_path text,
  created_at timestamptz DEFAULT now(),
  completed_at timestamptz,
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE
);

ALTER TABLE experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own experiments"
  ON experiments FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create experiments"
  ON experiments FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own experiments"
  ON experiments FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_experiments_user_id ON experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_exp_id ON experiments(exp_id);

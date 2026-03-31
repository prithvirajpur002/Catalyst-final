import { useState, useEffect } from 'react';
import { Play, CheckCircle, AlertCircle, TrendingUp } from 'lucide-react';
import { supabase } from '../lib/supabase';

interface ExperimentRun {
  id: string;
  exp_id: string;
  dataset: string;
  config: string;
  epochs: number;
  batch_size: number;
  status: 'pending' | 'running' | 'complete' | 'failed';
  scores?: Record<string, number>;
  created_at: string;
  completed_at?: string;
}

interface Decision {
  id: string;
  winner_exp_id: string;
  loser_exp_id?: string;
  reason_summary: string;
  next_planned_exp_id?: string;
  rationale?: string;
  created_at: string;
}

export default function ExperimentRunner() {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);
  const [newExpId, setNewExpId] = useState('exp_001');
  const [newDataset, setNewDataset] = useState('clean');
  const [newConfig, setNewConfig] = useState('baseline');
  const [newEpochs, setNewEpochs] = useState(200);
  const [newBatchSize, setNewBatchSize] = useState(6);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  async function loadData() {
    try {
      const { data: runsData, error: runsError } = await supabase
        .from('experiment_runs')
        .select('*')
        .order('created_at', { ascending: false });

      const { data: decisionsData, error: decisionsError } = await supabase
        .from('decisions')
        .select('*')
        .order('created_at', { ascending: false });

      if (runsError) throw runsError;
      if (decisionsError) throw decisionsError;

      setRuns(runsData || []);
      setDecisions(decisionsData || []);
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setLoading(false);
    }
  }

  async function createAndRunExperiment() {
    setError('');
    setSuccessMsg('');

    if (!newExpId.startsWith('exp_')) {
      setError('Experiment ID must start with "exp_"');
      return;
    }

    if (runs.some((r) => r.exp_id === newExpId)) {
      setError(`Experiment ${newExpId} already exists`);
      return;
    }

    try {
      const { error: insertError } = await supabase.from('experiment_runs').insert({
        exp_id: newExpId,
        dataset: newDataset,
        config: newConfig,
        epochs: newEpochs,
        batch_size: newBatchSize,
        status: 'pending',
      });

      if (insertError) throw insertError;

      setSuccessMsg(`Created ${newExpId}. Ready to run.`);
      setNewExpId(`exp_${(parseInt(newExpId.split('_')[1]) + 1).toString().padStart(3, '0')}`);

      await loadData();
    } catch (err) {
      setError(`Failed to create experiment: ${err}`);
    }
  }

  async function recordDecision(
    winnerExpId: string,
    loserExpId: string | undefined,
    reason: string
  ) {
    try {
      const { error } = await supabase.from('decisions').insert({
        winner_exp_id: winnerExpId,
        loser_exp_id: loserExpId,
        reason_summary: reason,
      });

      if (error) throw error;
      setSuccessMsg('Decision recorded.');
      await loadData();
    } catch (err) {
      setError(`Failed to record decision: ${err}`);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'complete':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  }

  function getCompositeScore(scores?: Record<string, number>): number {
    if (!scores) return 0;
    return (
      (scores.naturalness || 0) * 0.45 +
      (scores.clarity || 0) * 0.35 +
      (scores.identity || 0) * 0.20
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold text-slate-900 mb-2">Experiment Control</h1>
        <p className="text-slate-600 mb-8">
          Manual, traceable experiment management. One variable per experiment.
        </p>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex gap-3">
            <AlertCircle className="text-red-600 flex-shrink-0" size={20} />
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {successMsg && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex gap-3">
            <CheckCircle className="text-green-600 flex-shrink-0" size={20} />
            <p className="text-green-800">{successMsg}</p>
          </div>
        )}

        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Define New Experiment</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Experiment ID
              </label>
              <input
                type="text"
                value={newExpId}
                onChange={(e) => setNewExpId(e.target.value)}
                placeholder="exp_001"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Dataset
              </label>
              <select
                value={newDataset}
                onChange={(e) => setNewDataset(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              >
                <option value="clean">clean</option>
                <option value="natural">natural</option>
                <option value="raw">raw</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Config
              </label>
              <select
                value={newConfig}
                onChange={(e) => setNewConfig(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              >
                <option value="baseline">baseline</option>
                <option value="high_quality">high_quality</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Epochs
              </label>
              <input
                type="number"
                value={newEpochs}
                onChange={(e) => setNewEpochs(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                value={newBatchSize}
                onChange={(e) => setNewBatchSize(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900"
              />
            </div>
          </div>

          <button
            onClick={createAndRunExperiment}
            className="w-full bg-slate-900 hover:bg-slate-800 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <Play size={16} />
            Create Experiment
          </button>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-slate-900 mb-4 flex items-center gap-2">
            <TrendingUp size={20} />
            Experiment Runs
          </h2>

          {loading ? (
            <p className="text-slate-500">Loading...</p>
          ) : runs.length === 0 ? (
            <p className="text-slate-500">No experiments yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">
                      ID
                    </th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">
                      Dataset
                    </th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">
                      Config
                    </th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">
                      Params
                    </th>
                    <th className="px-4 py-2 text-left font-semibold text-slate-900">
                      Status
                    </th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-900">
                      Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr key={run.id} className="border-b border-slate-100 hover:bg-slate-50">
                      <td className="px-4 py-2 font-mono text-slate-900">{run.exp_id}</td>
                      <td className="px-4 py-2 text-slate-600">{run.dataset}</td>
                      <td className="px-4 py-2 text-slate-600">{run.config}</td>
                      <td className="px-4 py-2 text-xs text-slate-600">
                        {run.epochs}e, b{run.batch_size}
                      </td>
                      <td className="px-4 py-2">
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded border ${getStatusColor(
                            run.status
                          )}`}
                        >
                          {run.status}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right font-bold text-slate-900">
                        {run.scores && run.status === 'complete'
                          ? getCompositeScore(run.scores).toFixed(3)
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6">
          <h2 className="text-xl font-semibold text-slate-900 mb-4">Decisions</h2>

          {loading ? (
            <p className="text-slate-500">Loading...</p>
          ) : decisions.length === 0 ? (
            <p className="text-slate-500">No decisions yet. Compare experiments and record winners.</p>
          ) : (
            <div className="space-y-3">
              {decisions.map((decision) => (
                <div key={decision.id} className="p-3 bg-slate-50 rounded border border-slate-200">
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-semibold text-slate-900">
                      Winner: {decision.winner_exp_id}
                    </span>
                    <span className="text-xs text-slate-500">
                      {new Date(decision.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-700">{decision.reason_summary}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

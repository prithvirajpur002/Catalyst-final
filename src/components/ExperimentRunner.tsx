import { useState, useEffect } from 'react';
import { Play, RotateCcw, TrendingUp, Zap } from 'lucide-react';
import { supabase } from '../lib/supabase';

interface Experiment {
  id: string;
  exp_id: string;
  dataset: string;
  config: string;
  epochs: number;
  batch_size: number;
  mode: string;
  status: 'pending' | 'running' | 'complete' | 'failed';
  scores?: {
    naturalness?: number;
    clarity?: number;
    identity?: number;
    composite?: number;
  };
  created_at: string;
  completed_at?: string;
}

interface ExperimentConfig {
  id: string;
  dataset: 'clean' | 'natural';
  config: 'baseline' | 'high_quality';
  epochs: number;
  batch_size: number;
}

export default function ExperimentRunner() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);

  const presetConfigs: ExperimentConfig[] = [
    {
      id: 'exp_001',
      dataset: 'clean',
      config: 'baseline',
      epochs: 200,
      batch_size: 6,
    },
    {
      id: 'exp_002',
      dataset: 'natural',
      config: 'baseline',
      epochs: 200,
      batch_size: 6,
    },
    {
      id: 'exp_003',
      dataset: 'natural',
      config: 'baseline',
      epochs: 300,
      batch_size: 6,
    },
  ];

  useEffect(() => {
    loadExperiments();
    const interval = setInterval(loadExperiments, 5000);
    return () => clearInterval(interval);
  }, []);

  async function loadExperiments() {
    try {
      const { data, error } = await supabase
        .from('experiments')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setExperiments(data || []);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    } finally {
      setLoading(false);
    }
  }

  async function startExperiment(config: ExperimentConfig) {
    setRunning(config.id);
    try {
      const { error: insertError } = await supabase.from('experiments').insert({
        exp_id: config.id,
        dataset: config.dataset,
        config: config.config,
        epochs: config.epochs,
        batch_size: config.batch_size,
        mode: config.dataset,
        status: 'running',
      });

      if (insertError) throw insertError;

      await loadExperiments();
    } catch (err) {
      console.error('Failed to start experiment:', err);
    } finally {
      setRunning(null);
    }
  }

  function getCompositeScore(exp: Experiment): number {
    if (!exp.scores) return 0;
    return (
      (exp.scores.naturalness || 0) * 0.45 +
      (exp.scores.clarity || 0) * 0.35 +
      (exp.scores.identity || 0) * 0.20
    );
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'complete':
        return 'bg-green-100 text-green-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">
            RVC Experiment Runner
          </h1>
          <p className="text-slate-600">
            Run controlled experiments with different configurations and compare results.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {presetConfigs.map((config) => {
            const existingExp = experiments.find((e) => e.exp_id === config.id);
            const isRunning = running === config.id;

            return (
              <div
                key={config.id}
                className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-slate-900">
                    {config.id}
                  </h3>
                  <p className="text-sm text-slate-500 mt-1">
                    {config.dataset} mode
                  </p>
                </div>

                <div className="space-y-2 mb-4 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Config:</span>
                    <span className="font-medium text-slate-900">
                      {config.config}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Epochs:</span>
                    <span className="font-medium text-slate-900">
                      {config.epochs}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600">Batch:</span>
                    <span className="font-medium text-slate-900">
                      {config.batch_size}
                    </span>
                  </div>
                </div>

                {existingExp && (
                  <div className="mb-4 p-3 bg-slate-50 rounded border border-slate-100">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-slate-600">
                        STATUS
                      </span>
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded ${getStatusColor(
                          existingExp.status
                        )}`}
                      >
                        {existingExp.status}
                      </span>
                    </div>
                    {existingExp.scores && existingExp.status === 'complete' && (
                      <div className="space-y-1 mt-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-600">Naturalness:</span>
                          <span className="font-semibold text-slate-900">
                            {(existingExp.scores.naturalness || 0).toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-600">Clarity:</span>
                          <span className="font-semibold text-slate-900">
                            {(existingExp.scores.clarity || 0).toFixed(3)}
                          </span>
                        </div>
                        <div className="flex justify-between pt-1 border-t border-slate-200">
                          <span className="text-slate-700 font-semibold">
                            Composite:
                          </span>
                          <span className="text-slate-900 font-bold">
                            {getCompositeScore(existingExp).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <button
                  onClick={() => startExperiment(config)}
                  disabled={isRunning || existingExp?.status === 'running'}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
                >
                  {isRunning || existingExp?.status === 'running' ? (
                    <>
                      <RotateCcw size={16} className="animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play size={16} />
                      {existingExp ? 'Re-run' : 'Start'}
                    </>
                  )}
                </button>
              </div>
            );
          })}
        </div>

        <div className="bg-white rounded-lg border border-slate-200 shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp size={20} />
              Experiment Results
            </h2>
          </div>

          {loading ? (
            <div className="px-6 py-8 text-center text-slate-500">
              Loading experiments...
            </div>
          ) : experiments.length === 0 ? (
            <div className="px-6 py-8 text-center text-slate-500">
              No experiments yet. Start one above to begin.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3 text-left font-semibold text-slate-900">
                      Experiment
                    </th>
                    <th className="px-6 py-3 text-left font-semibold text-slate-900">
                      Config
                    </th>
                    <th className="px-6 py-3 text-left font-semibold text-slate-900">
                      Mode
                    </th>
                    <th className="px-6 py-3 text-left font-semibold text-slate-900">
                      Status
                    </th>
                    <th className="px-6 py-3 text-right font-semibold text-slate-900">
                      Composite Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((exp) => (
                    <tr
                      key={exp.id}
                      className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
                    >
                      <td className="px-6 py-3 font-semibold text-slate-900">
                        {exp.exp_id}
                      </td>
                      <td className="px-6 py-3 text-slate-600">{exp.config}</td>
                      <td className="px-6 py-3 text-slate-600">{exp.mode}</td>
                      <td className="px-6 py-3">
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded ${getStatusColor(
                            exp.status
                          )}`}
                        >
                          {exp.status}
                        </span>
                      </td>
                      <td className="px-6 py-3 text-right font-bold text-slate-900">
                        {exp.scores && exp.status === 'complete' ? (
                          <span className="flex items-center justify-end gap-1">
                            <Zap size={14} className="text-amber-500" />
                            {getCompositeScore(exp).toFixed(3)}
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

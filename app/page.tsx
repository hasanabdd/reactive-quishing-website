'use client';

import React, {
  useState,
  useEffect,
  useRef,
  FormEvent,
  DragEvent,
  ChangeEvent,
} from 'react';

type View = 'home' | 'upload' | 'link' | 'loading' | 'result' | 'dashboard';
type ResultType = 'safe' | 'suspicious' | 'malicious';

const classifyUrl = (rawUrl: string): ResultType => {
  const url = rawUrl.toLowerCase();

  const maliciousTlds = ['.ru', '.tk', '.cn'];
  const suspiciousWords = ['free', 'promo', 'offer', 'gift', 'win', 'login', 'verify'];
  const phishingHints = ['bank', 'paypal', 'appleid', 'office365'];

  const hasBadTld = maliciousTlds.some((tld) => url.includes(tld));
  const hasSuspiciousWord = suspiciousWords.some((w) => url.includes(w));
  const hasPhishingHint = phishingHints.some((w) => url.includes(w));
  const isHttpOnly = url.startsWith('http://') && !url.startsWith('https://');

  if (hasBadTld || (hasPhishingHint && (hasSuspiciousWord || isHttpOnly))) {
    return 'malicious';
  }

  if (hasSuspiciousWord || isHttpOnly || url.length > 70) {
    return 'suspicious';
  }

  return 'safe';
};

const resultCopy: Record<
  ResultType,
  {
    title: string;
    icon: string;
    colorClasses: string;
    buttonClasses: string;
    descriptionList: { icon: string; title: string; desc: string }[];
  }
> = {
  safe: {
    title: 'Safe to Open',
    icon: '✅',
    colorClasses: 'text-emerald-400',
    buttonClasses:
      'bg-gradient-to-r from-emerald-500 to-emerald-600 shadow-emerald-500/40',
    descriptionList: [
      {
        icon: '🔒',
        title: 'HTTPS Enabled',
        desc: 'Secure connection verified with valid certificate.',
      },
      {
        icon: '✓',
        title: 'Domain Verified',
        desc: 'No known malicious history for this domain.',
      },
      {
        icon: '🌐',
        title: 'No Threats Detected',
        desc: 'No phishing or malware indicators found.',
      },
    ],
  },
  suspicious: {
    title: 'Suspicious Link',
    icon: '⚠️',
    colorClasses: 'text-amber-400',
    buttonClasses:
      'bg-gradient-to-r from-amber-500 to-amber-600 shadow-amber-500/40',
    descriptionList: [
      {
        icon: '⚠️',
        title: 'Unusual Pattern',
        desc: 'The URL contains patterns often seen in scams or spam.',
      },
      {
        icon: '🔍',
        title: 'Requires Caution',
        desc: 'Limited reputation data or recently registered domain.',
      },
      {
        icon: '📊',
        title: 'Low Trust Score',
        desc: 'We recommend double-checking the sender before opening.',
      },
    ],
  },
  malicious: {
    title: 'Malicious Link Detected',
    icon: '🚫',
    colorClasses: 'text-red-400',
    buttonClasses:
      'bg-gradient-to-r from-red-500 to-red-600 shadow-red-500/40',
    descriptionList: [
      {
        icon: '🎣',
        title: 'Phishing Attempt',
        desc: 'This link likely tries to steal credentials or data.',
      },
      {
        icon: '🦠',
        title: 'Malware Risk',
        desc: 'Known malicious patterns detected in this URL.',
      },
      {
        icon: '🚫',
        title: 'Blacklisted Indicators',
        desc: 'The URL matches patterns used in known attacks.',
      },
    ],
  },
};

export default function HomePage() {
  const [view, setView] = useState<View>('home');
  const [demoResult, setDemoResult] = useState<ResultType>('safe');
  const [resultType, setResultType] = useState<ResultType | null>(null);
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [urlInput, setUrlInput] = useState<string>('');
  const [toast, setToast] = useState<string | null>(null);

  // camera state
  const [showCamera, setShowCamera] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (!showCamera) {
      stopCamera();
      return;
    }

    const start = async () => {
      if (typeof navigator === 'undefined' || !navigator.mediaDevices) {
        setCameraError('Camera is not supported in this environment.');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setCameraError(null);
      } catch (err) {
        console.error(err);
        setCameraError('Unable to access camera. Please check permissions.');
      }
    };

    start();

    return () => {
      stopCamera();
    };
  }, [showCamera]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  const showToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 2500);
  };

  const goHome = () => {
    setView('home');
    setResultType(null);
    setCurrentUrl('');
  };

  const startFakeScan = (urlFrom?: string) => {
    setView('loading');
    const fakeUrl = urlFrom ?? 'https://example.com/from-qr';
    setCurrentUrl(fakeUrl);

    setTimeout(() => {
      setResultType(demoResult);
      setView('result');
    }, 1300);
  };

  const handleFile = (file: File | null) => {
    if (!file) return;
    showToast(`Scanning "${file.name}" as QR code…`);
    startFakeScan();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    handleFile(file);
  };

  const handleUrlSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!urlInput.trim()) {
      showToast('Please paste a URL to analyze.');
      return;
    }
    const type = classifyUrl(urlInput.trim());
    setCurrentUrl(urlInput.trim());
    setResultType(type);
    setView('result');
  };

  const handlePaste = async () => {
    try {
      if (navigator.clipboard && navigator.clipboard.readText) {
        const text = await navigator.clipboard.readText();
        if (text) {
          setUrlInput(text);
          showToast('Pasted URL from clipboard.');
        } else {
          showToast('Clipboard is empty.');
        }
      } else {
        showToast('Clipboard API not available.');
      }
    } catch {
      showToast('Unable to access clipboard.');
    }
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-50 overflow-hidden">
      {/* Soft radial glows */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute -top-24 -left-24 h-72 w-72 rounded-full bg-blue-500/10 blur-3xl" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-purple-500/10 blur-3xl" />
        <div className="absolute top-1/3 left-1/2 h-60 w-60 -translate-x-1/2 rounded-full bg-emerald-400/10 blur-3xl" />
      </div>

      {/* Nav */}
      <nav className="fixed inset-x-0 top-0 z-30 flex items-center justify-between border-b border-slate-800/60 bg-slate-950/40 px-6 py-3 backdrop-blur-xl">
        <button
          onClick={goHome}
          className="flex items-center gap-2 text-lg font-semibold text-slate-50 hover:scale-[1.02] transition-transform"
        >
          <span className="text-2xl">🛡️</span>
          <span>Quishing Detector</span>
        </button>

        <div className="flex items-center gap-2 rounded-2xl border border-slate-700/80 bg-slate-900/70 px-3 py-2 text-xs sm:text-sm">
          <span className="mr-1 text-slate-300">Demo result:</span>
          {(['safe', 'suspicious', 'malicious'] as ResultType[]).map((r) => (
            <button
              key={r}
              onClick={() => setDemoResult(r)}
              className={[
                'rounded-xl px-2.5 py-1 font-semibold transition-all',
                demoResult === r
                  ? r === 'safe'
                    ? 'bg-emerald-500/90 text-white shadow-lg shadow-emerald-500/40'
                    : r === 'suspicious'
                    ? 'bg-amber-500/90 text-white shadow-lg shadow-amber-500/40'
                    : 'bg-red-500/90 text-white shadow-lg shadow-red-500/40'
                  : 'bg-slate-800/70 text-slate-200 hover:bg-slate-700',
              ].join(' ')}
            >
              {r === 'safe' && '✅ Safe'}
              {r === 'suspicious' && '⚠️ Suspicious'}
              {r === 'malicious' && '🚫 Malicious'}
            </button>
          ))}
        </div>
      </nav>

      {/* Main container */}
      <main className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col px-4 pb-20 pt-24 sm:px-6 lg:px-8">
        {view === 'home' && (
          <section className="flex flex-1 flex-col items-center justify-center gap-12 text-center">
            <div className="max-w-2xl space-y-4">
              <div className="mx-auto mb-4 inline-flex items-center justify-center rounded-2xl bg-slate-900/80 px-3 py-1 text-xs font-medium text-slate-300 ring-1 ring-slate-700/80">
                <span className="mr-2 text-lg">✨</span>
                Real-time protection from quishing & phishing links
              </div>

              <div className="text-6xl mb-4 drop-shadow-[0_0_30px_rgba(56,189,248,0.35)]">
                🛡️
              </div>

              <h1 className="bg-gradient-to-r from-slate-50 via-sky-100 to-indigo-200 bg-clip-text text-3xl font-extrabold leading-tight text-transparent sm:text-4xl lg:text-5xl">
                Protect yourself from malicious QR codes & phishing links
              </h1>

              <p className="text-sm text-slate-200/90 sm:text-base">
                Scan QR codes, paste URLs, and simulate how your security engine
                would react. Perfect for demos, presentations, and your LOCO
                project prototype.
              </p>

              <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
                <button
                  onClick={() => setView('upload')}
                  className="relative inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500 px-5 py-2.5 text-sm font-semibold text-white shadow-xl shadow-sky-500/40 transition hover:-translate-y-[2px] hover:shadow-2xl"
                >
                  <span>📷 Scan QR Code</span>
                </button>
                <button
                  onClick={() => setView('link')}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/70 px-5 py-2.5 text-sm font-semibold text-slate-100 transition hover:-translate-y-[2px] hover:bg-slate-800"
                >
                  <span>🔗 Check a Link</span>
                </button>
                <button
                  onClick={() => setView('dashboard')}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-900/60 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800"
                >
                  <span>📊 View demo dashboard</span>
                </button>
              </div>
            </div>

            {/* Features */}
            <div className="grid w-full max-w-5xl gap-4 md:grid-cols-2 lg:grid-cols-4">
              {[
                {
                  icon: '🔍',
                  title: 'Advanced detection',
                  text: 'Classifies URLs into safe, suspicious, or malicious instantly.',
                },
                {
                  icon: '🔗',
                  title: 'Multi-method scanning',
                  text: 'Simulate QR scans, link pastes, and uploads in one interface.',
                },
                {
                  icon: '⚡',
                  title: 'Instant feedback',
                  text: 'Smooth transitions and animated results for clear demos.',
                },
                {
                  icon: '🔒',
                  title: 'Privacy-friendly demo',
                  text: 'Everything runs in the browser; no real data sent anywhere.',
                },
              ].map((f) => (
                <div
                  key={f.title}
                  className="group rounded-2xl border border-slate-800 bg-slate-900/70 p-4 text-left shadow-lg shadow-slate-950/70 transition hover:-translate-y-1 hover:border-sky-500/70 hover:shadow-sky-500/30"
                >
                  <div className="mb-3 text-2xl drop-shadow-lg">{f.icon}</div>
                  <h3 className="mb-1 text-sm font-semibold text-slate-50">
                    {f.title}
                  </h3>
                  <p className="text-xs text-slate-300">{f.text}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {view === 'upload' && (
          <section className="flex flex-1 flex-col items-center justify-center">
            <div className="w-full max-w-xl rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl shadow-slate-950/70 backdrop-blur-xl">
              <h2 className="mb-1 text-center text-2xl font-bold">
                Scan QR Code
              </h2>
              <p className="mb-5 text-center text-sm text-slate-300">
                Upload any image to simulate a QR code scan and see how the
                engine reacts.
              </p>

              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700 bg-slate-950/50 px-6 py-8 text-center transition hover:border-sky-500/80 hover:bg-slate-900/60"
              >
                <div className="mb-3 text-4xl">📷</div>
                <p className="text-sm font-medium text-slate-100">
                  Drag & drop your QR image here
                </p>
                <p className="mt-1 text-xs text-slate-400">or</p>
                <div className="mt-3 flex flex-wrap justify-center gap-3">
                  <button
                    type="button"
                    onClick={() => setShowCamera(true)}
                    className="rounded-xl bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-50 hover:bg-slate-700"
                  >
                    📸 Use camera
                  </button>
                  <label className="cursor-pointer rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-sky-500/40 hover:-translate-y-[1px] hover:shadow-2xl">
                    📁 Choose file
                    <input
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleFileInput}
                    />
                  </label>
                </div>
              </div>

              <p className="mt-3 text-center text-[11px] text-slate-400">
                💡 This is a demo. Any image will be treated as a QR code and
                classified based on the selected demo result.
              </p>

              <div className="mt-5 flex justify-center">
                <button
                  onClick={goHome}
                  className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-100 hover:bg-slate-800"
                >
                  ← Back to home
                </button>
              </div>
            </div>
          </section>
        )}

        {view === 'link' && (
          <section className="flex flex-1 flex-col items-center justify-center">
            <div className="w-full max-w-xl rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-slate-950/80 backdrop-blur-xl">
              <h2 className="mb-1 text-center text-2xl font-bold">
                🔗 Check a link
              </h2>
              <p className="mb-5 text-center text-sm text-slate-300">
                Paste any URL and let the client-side engine classify it.
              </p>

              <form onSubmit={handleUrlSubmit} className="space-y-3">
                <label className="block text-xs font-semibold text-slate-200">
                  URL to analyze
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://example.com/suspicious-link"
                    className="w-full rounded-xl border border-slate-400/70 bg-slate-100 px-3 py-2 pr-16 text-xs text-slate-900 outline-none ring-2 ring-transparent transition focus:border-indigo-500 focus:ring-indigo-500/40"
                  />
                  <button
                    type="button"
                    onClick={handlePaste}
                    className="absolute right-1.5 top-1.5 rounded-lg bg-slate-200 px-2 py-1 text-[11px] font-semibold text-slate-700 hover:bg-slate-300"
                  >
                    📋 Paste
                  </button>
                </div>

                <button
                  type="submit"
                  className="mt-2 w-full rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-sky-500/40 hover:-translate-y-[1px] hover:shadow-2xl"
                >
                  🔍🛡️ Analyze link
                </button>
              </form>

              <div className="mt-4 rounded-2xl bg-slate-950/70 p-3 text-xs text-slate-200 ring-1 ring-slate-700/70">
                <div className="mb-1 flex items-center gap-1 font-semibold">
                  <span>ℹ️</span>
                  <span>Heuristics used in this demo:</span>
                </div>
                <ul className="ml-4 list-disc space-y-0.5 text-[11px] text-slate-300">
                  <li>Dangerous or uncommon TLDs (.ru, .tk, etc.)</li>
                  <li>Suspicious words like &quot;free&quot;, &quot;offer&quot;, &quot;login&quot;</li>
                  <li>Non-HTTPS links and overly long URLs</li>
                  <li>Phishing-style keywords like &quot;bank&quot; or &quot;paypal&quot;</li>
                </ul>
              </div>

              <div className="mt-4 flex justify-center">
                <button
                  onClick={goHome}
                  className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-100 hover:bg-slate-800"
                >
                  ← Back to home
                </button>
              </div>
            </div>
          </section>
        )}

        {view === 'loading' && (
          <section className="flex flex-1 flex-col items-center justify-center">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="text-6xl animate-spin drop-shadow-[0_0_40px_rgba(56,189,248,0.7)]">
                🛡️
              </div>
              <h2 className="bg-gradient-to-r from-slate-50 to-sky-100 bg-clip-text text-2xl font-bold text-transparent">
                Extracting URL…
              </h2>
              <p className="text-sm text-slate-300">
                Simulating scan and risk analysis. This is a front-end demo
                only.
              </p>
              <div className="mt-2 flex gap-2">
                <span className="h-2 w-2 animate-[ping_1.2s_infinite] rounded-full bg-sky-400" />
                <span className="h-2 w-2 animate-[ping_1.2s_infinite_200ms] rounded-full bg-indigo-400" />
                <span className="h-2 w-2 animate-[ping_1.2s_infinite_400ms] rounded-full bg-emerald-400" />
              </div>
            </div>
          </section>
        )}

        {view === 'result' && resultType && (
          <section className="flex flex-1 flex-col items-center justify-center">
            <ResultSection
              type={resultType}
              url={currentUrl}
              onScanAnother={() => setView('home')}
            />
          </section>
        )}

        {view === 'dashboard' && (
          <section className="flex flex-1 flex-col">
            <div className="mb-6 text-center">
              <h2 className="bg-gradient-to-r from-slate-50 to-sky-200 bg-clip-text text-3xl font-extrabold text-transparent">
                Security dashboard
              </h2>
              <p className="mt-1 text-sm text-slate-300">
                A demo view of how LOCO could track scans and threats over time.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-4">
              {[
                { icon: '📊', label: 'Total scans', value: '12,847' },
                { icon: '✅', label: 'Safe links', value: '10,234' },
                { icon: '⚠️', label: 'Suspicious links', value: '1,892' },
                { icon: '🚫', label: 'Threats blocked', value: '721' },
              ].map((s) => (
                <div
                  key={s.label}
                  className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg shadow-slate-950/70"
                >
                  <div className="mb-2 text-3xl drop-shadow">{s.icon}</div>
                  <div className="text-2xl font-extrabold text-slate-50">
                    {s.value}
                  </div>
                  <div className="text-xs font-medium text-slate-300">
                    {s.label}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg shadow-slate-950/70">
              <h3 className="mb-3 text-sm font-semibold text-slate-100">
                Recent scans (demo data)
              </h3>
              <div className="overflow-x-auto text-xs">
                <table className="min-w-full border-separate border-spacing-y-1">
                  <thead className="text-[11px] uppercase text-slate-400">
                    <tr>
                      <th className="pb-2 text-left">URL</th>
                      <th className="pb-2 text-left">Status</th>
                      <th className="pb-2 text-left">Time</th>
                    </tr>
                  </thead>
                  <tbody className="text-[11px]">
                    {[
                      {
                        url: 'example-shop.com',
                        status: 'Safe',
                        type: 'safe',
                        time: '2 min ago',
                      },
                      {
                        url: 'suspicious-promo.xyz',
                        status: 'Suspicious',
                        type: 'suspicious',
                        time: '5 min ago',
                      },
                      {
                        url: 'phishing-bank.ru',
                        status: 'Malicious',
                        type: 'malicious',
                        time: '8 min ago',
                      },
                      {
                        url: 'company-website.com',
                        status: 'Safe',
                        type: 'safe',
                        time: '12 min ago',
                      },
                      {
                        url: 'free-prize-claim.tk',
                        status: 'Malicious',
                        type: 'malicious',
                        time: '15 min ago',
                      },
                    ].map((row) => (
                      <tr key={row.url}>
                        <td className="rounded-l-xl bg-slate-950/60 px-3 py-2">
                          {row.url}
                        </td>
                        <td className="bg-slate-950/60 px-3 py-2">
                          <span
                            className={[
                              'inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold',
                              row.type === 'safe'
                                ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
                                : row.type === 'suspicious'
                                ? 'border-amber-500/50 bg-amber-500/10 text-amber-300'
                                : 'border-red-500/50 bg-red-500/10 text-red-300',
                            ].join(' ')}
                          >
                            {row.status}
                          </span>
                        </td>
                        <td className="rounded-r-xl bg-slate-950/60 px-3 py-2 text-slate-300">
                          {row.time}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-6 flex justify-center">
              <button
                onClick={goHome}
                className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-100 hover:bg-slate-800"
              >
                ← Back to home
              </button>
            </div>
          </section>
        )}
      </main>

      {/* Camera modal */}
      {showCamera && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-md rounded-3xl border border-slate-700 bg-slate-900/95 p-4 shadow-2xl">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-100">
                📸 Scan QR code (demo)
              </h3>
              <button
                onClick={() => setShowCamera(false)}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-red-500/10 text-sm text-red-400 hover:bg-red-500/20"
              >
                ×
              </button>
            </div>

            <div className="overflow-hidden rounded-2xl border border-slate-700 bg-black">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="block h-56 w-full object-cover"
              />
            </div>

            {cameraError && (
              <p className="mt-2 text-[11px] text-red-300">{cameraError}</p>
            )}

            <div className="mt-4 flex gap-2">
              <button
                onClick={() => {
                  setShowCamera(false);
                  startFakeScan('https://example.com/from-camera');
                }}
                className="flex-1 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-500 px-3 py-2 text-xs font-semibold text-white shadow-lg shadow-sky-500/40 hover:-translate-y-[1px] hover:shadow-2xl"
              >
                📸 Capture & simulate scan
              </button>
            </div>

            <p className="mt-2 text-[10px] text-slate-400">
              This is a front-end demo only. No frames are uploaded; we just
              trigger the same classification flow.
            </p>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-2xl border border-slate-700 bg-slate-900/95 px-4 py-2 text-xs font-medium text-slate-100 shadow-xl">
          {toast}
        </div>
      )}

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-800/80 bg-slate-950/70 py-4 text-center text-[11px] text-slate-400">
        <p className="font-semibold text-slate-200">
          Copyright © 2025 Quishing Detection
        </p>
        <p>Built for security demos & quishing awareness.</p>
        <p className="italic text-slate-500">
          By the local, for the local 🧠
        </p>
      </footer>
    </div>
  );
}

type ResultProps = {
  type: ResultType;
  url: string;
  onScanAnother: () => void;
};

function ResultSection({ type, url, onScanAnother }: ResultProps) {
  const cfg = resultCopy[type];

  return (
    <div className="w-full max-w-xl rounded-3xl border border-slate-800 bg-slate-900/80 p-6 text-center shadow-2xl shadow-slate-950/80 backdrop-blur-xl">
      <div className="mb-3 text-5xl drop-shadow-[0_0_30px_rgba(56,189,248,0.6)]">
        {cfg.icon}
      </div>
      <h2
        className={[
          'text-2xl font-extrabold drop-shadow-sm',
          cfg.colorClasses,
        ].join(' ')}
      >
        {cfg.title}
      </h2>

      <div className="mt-4 rounded-xl bg-slate-950/70 p-3 text-left text-[11px] font-mono text-slate-100 ring-1 ring-slate-700/70">
        {url || 'https://example.com'}
      </div>

      <div className="mt-4 space-y-2 text-left text-xs">
        {cfg.descriptionList.map((d) => (
          <div
            key={d.title}
            className="flex gap-2 rounded-xl bg-slate-950/70 p-3 ring-1 ring-slate-800"
          >
            <div className="text-lg">{d.icon}</div>
            <div>
              <div className="font-semibold text-slate-100">{d.title}</div>
              <div className="text-[11px] text-slate-300">{d.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-4 rounded-xl bg-slate-950/70 px-3 py-2 text-[10px] text-slate-400">
        ℹ️ This is a simulated front-end result for demonstration purposes. In
        your final LOCO app, these scores would come from your ML model and
        backend.
      </p>

      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {type === 'safe' && (
          <a
            href={url || '#'}
            target="_blank"
            rel="noreferrer"
            className={[
              'inline-flex items-center justify-center rounded-xl px-4 py-2 text-xs font-semibold text-white shadow-lg',
              cfg.buttonClasses,
            ].join(' ')}
          >
            Open URL in new tab
          </a>
        )}
        {type !== 'safe' && (
          <button
            disabled
            className={[
              'inline-flex cursor-not-allowed items-center justify-center rounded-xl px-4 py-2 text-xs font-semibold text-white opacity-80',
              cfg.buttonClasses,
            ].join(' ')}
          >
            Do not open (blocked)
          </button>
        )}
        <button
          onClick={onScanAnother}
          className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-semibold text-slate-100 hover:bg-slate-800"
        >
          Scan another
        </button>
      </div>
    </div>
  );
}

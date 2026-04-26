/* ── Meta-OpenENV Dashboard — Particles, Animations, Live Demo, Training ── */

document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initScrollReveal();
  initNavScroll();
  initCounters();
  initGraphModal();
});

/* ── Particle Background ────────────────────────────────────────────────── */
function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let particles = [];
  const N = 60;

  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);

  for (let i = 0; i < N; i++) {
    particles.push({
      x: Math.random() * canvas.width, y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 2 + 0.5,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((p, i) => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(79, 140, 255, 0.4)'; ctx.fill();
      for (let j = i + 1; j < particles.length; j++) {
        const dx = p.x - particles[j].x, dy = p.y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(79, 140, 255, ${0.15 * (1 - dist / 150)})`;
          ctx.lineWidth = 0.5; ctx.stroke();
        }
      }
    });
    requestAnimationFrame(draw);
  }
  draw();
}

/* ── Scroll Reveal ──────────────────────────────────────────────────────── */
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        entry.target.querySelectorAll('.bar-fill[data-width]').forEach(bar => {
          bar.style.width = bar.dataset.width;
        });
      }
    });
  }, { threshold: 0.15 });
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

/* ── Navbar scroll effect ───────────────────────────────────────────────── */
function initNavScroll() {
  const nav = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 50);
  });
}

/* ── Animated Counters ──────────────────────────────────────────────────── */
function initCounters() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.dataset.counted) {
        entry.target.dataset.counted = 'true';
        animateCounter(entry.target);
      }
    });
  }, { threshold: 0.5 });
  document.querySelectorAll('[data-count]').forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target = parseFloat(el.dataset.count);
  const suffix = el.dataset.suffix || '';
  const prefix = el.dataset.prefix || '';
  const decimals = (el.dataset.count.includes('.')) ? el.dataset.count.split('.')[1].length : 0;
  const duration = 2000;
  const start = performance.now();
  function update(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ── Graph Lightbox Modal ───────────────────────────────────────────────── */
function initGraphModal() {
  const modal = document.getElementById('graph-modal');
  const modalImg = document.getElementById('modal-img');
  document.querySelectorAll('.graph-card img').forEach(img => {
    img.addEventListener('click', () => {
      modalImg.src = img.src; modal.classList.add('active');
    });
  });
  modal?.addEventListener('click', () => modal.classList.remove('active'));
}

/* ── Live API Demo ──────────────────────────────────────────────────────── */
async function runDemo() {
  const btn = document.getElementById('demo-btn');
  const reqOut = document.getElementById('demo-request');
  const resOut = document.getElementById('demo-response');
  const BASE = window.location.origin;

  btn.disabled = true; btn.textContent = '⏳ Running...';
  reqOut.textContent = ''; resOut.textContent = '';

  try {
    // Step 1: Reset
    const seed = Math.floor(Math.random() * 1000);
    const resetPayload = { task_id: 2, seed: seed };
    reqOut.textContent = `POST ${BASE}/reset\n${JSON.stringify(resetPayload, null, 2)}`;

    const resetRes = await fetch(`${BASE}/reset`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(resetPayload),
    });
    const session = await resetRes.json();
    const sid = session.session_id;
    const obs = session.observation || {};
    const ws = session.world_state || {};

    resOut.textContent =
      `✅ Session: ${sid}\n` +
      `📋 Policy: ${obs.active_policy_version || 'v1'}\n\n` +
      `🎫 Ticket: ${obs.subject || 'N/A'}\n` +
      `📝 ${(obs.body || '').substring(0, 120)}...\n` +
      `👤 Tier: ${obs.tier || 'N/A'}\n` +
      `💰 Balance: $${(ws.company_balance || 10000).toLocaleString()}\n` +
      `⚠️ SLA Breaches: ${ws.sla_breaches || 0}\n` +
      `📊 Difficulty: Level ${ws.difficulty_level || 1}`;

    // Step 2: Take action (using correct API format with nested action)
    const actionPayload = {
      session_id: sid,
      action: {
        assign_priority: 'High',
        assign_category: 'Technical',
        draft_response: 'Hello, thank you for reaching out regarding this issue. Our engineering team has been notified and is actively investigating. We will provide an update within 2 hours. Best regards, Support Team',
        escalate: false,
      }
    };

    reqOut.textContent +=
      `\n\n── Step 1 ──\n` +
      `POST ${BASE}/step\n` +
      JSON.stringify(actionPayload, null, 2);

    const stepRes = await fetch(`${BASE}/step`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(actionPayload),
    });
    const result = await stepRes.json();
    const stepWs = result.world_state || {};
    const nextObs = result.observation || {};

    resOut.textContent +=
      `\n\n── Step Result ──\n` +
      `🏆 Reward: ${typeof result.reward === 'number' ? result.reward.toFixed(3) : JSON.stringify(result.reward)}\n` +
      `✅ Done: ${result.done}\n` +
      `💰 Balance: $${(stepWs.company_balance || 0).toLocaleString()}\n` +
      `📉 Churn Risk: ${((stepWs.churn_risk || 0) * 100).toFixed(1)}%\n` +
      `⚠️ SLA Breaches: ${stepWs.sla_breaches || 0}\n` +
      `🎯 Attacker Win Rate: ${((stepWs.attacker_win_rate || 0) * 100).toFixed(1)}%\n` +
      `📊 Difficulty: Level ${stepWs.difficulty_level || 1}\n` +
      (result.drift_notice ? `\n🔔 Drift Notice: ${result.drift_notice}` : '') +
      (result.catastrophic ? '\n🚨 CATASTROPHIC FAILURE!' : '') +
      `\n\n── Next Ticket ──\n` +
      `🎫 ${nextObs.subject || 'Episode ended'}\n` +
      `👤 Tier: ${nextObs.tier || '-'}`;

  } catch (err) {
    resOut.textContent = `❌ Error: ${err.message}\n\nMake sure the API server is running at ${BASE}`;
  }

  btn.disabled = false; btn.textContent = '▶ Run Live Demo';
}

/* ── Run Multi-Step Demo ────────────────────────────────────────────────── */
async function runFullDemo() {
  const btn = document.getElementById('full-demo-btn');
  const output = document.getElementById('full-demo-output');
  const BASE = window.location.origin;

  btn.disabled = true; btn.textContent = '⏳ Running 12-step episode...';
  output.textContent = 'Starting episode...\n';

  try {
    const resetRes = await fetch(`${BASE}/reset`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: 2, seed: Math.floor(Math.random() * 10000) }),
    });
    const session = await resetRes.json();
    const sid = session.session_id;
    output.textContent += `✅ Session: ${sid}\n\n`;

    let totalReward = 0, steps = 0, done = false;
    let obs = session.observation;

    while (!done && steps < 12) {
      const text = ((obs.subject || '') + ' ' + (obs.body || '')).toLowerCase();
      let priority = 'Medium';
      if (/urgent|critical|emergency|down|outage|crash|500|fail/.test(text)) priority = 'Critical';
      else if (/cannot|locked|charged|error|bug|slow/.test(text)) priority = 'High';
      else if (/question|pricing|general|feedback|info/.test(text)) priority = 'Low';

      let category = 'Technical';
      if (/billing|charge|invoice|payment|refund|subscription/.test(text)) category = 'Billing';
      else if (/account|login|password|access|permission/.test(text)) category = 'Account';

      const stepRes = await fetch(`${BASE}/step`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sid,
          action: {
            assign_priority: priority, assign_category: category,
            draft_response: `Hello, thank you for contacting us about "${obs.subject}". Our team is investigating and will update you shortly. Best regards, Support`,
            escalate: priority === 'Critical',
          }
        }),
      });
      const result = await stepRes.json();
      totalReward += result.reward;
      steps++;
      done = result.done;

      const ws = result.world_state || {};
      output.textContent +=
        `Step ${steps}: ${obs.subject?.substring(0, 50)}...\n` +
        `  → Priority: ${priority} | Category: ${category} | Reward: ${result.reward.toFixed(2)}\n` +
        `  → Balance: $${ws.company_balance?.toLocaleString()} | SLA: ${ws.sla_breaches} | Churn: ${(ws.churn_risk * 100).toFixed(0)}%\n\n`;

      obs = result.observation || {};
      output.scrollTop = output.scrollHeight;
    }

    output.textContent +=
      `═══════════════════════════════\n` +
      `📊 Episode Complete!\n` +
      `   Steps: ${steps}\n` +
      `   Total Reward: ${totalReward.toFixed(2)}\n` +
      `   Mean Reward: ${(totalReward / steps).toFixed(3)}\n` +
      `═══════════════════════════════`;

  } catch (err) {
    output.textContent += `\n❌ Error: ${err.message}`;
  }

  btn.disabled = false; btn.textContent = '▶ Run Full 12-Step Episode';
}

/* ── Train & Generate Graphs ────────────────────────────────────────────── */
async function runTraining() {
  const btn = document.getElementById('train-btn');
  const status = document.getElementById('train-status');
  const BASE = window.location.origin;

  btn.disabled = true; btn.textContent = '⏳ Generating plots...';
  status.textContent = '🔄 Running 50-episode simulation and generating graphs...';
  status.style.color = '#f59e0b';

  try {
    const res = await fetch(`${BASE}/generate_plots`, { method: 'POST' });
    const data = await res.json();

    if (data.status === 'ok') {
      status.textContent = `✅ All ${data.plots_generated} plots generated in ${data.duration_ms}ms!`;
      status.style.color = '#06d6a0';

      // Reload all graph images with cache-busting
      const ts = Date.now();
      document.querySelectorAll('.graph-card img').forEach(img => {
        const src = img.src.split('?')[0];
        img.src = src + '?t=' + ts;
      });
    } else {
      status.textContent = `❌ Error: ${data.error}`;
      status.style.color = '#ef4444';
    }
  } catch (err) {
    status.textContent = `❌ Failed: ${err.message}`;
    status.style.color = '#ef4444';
  }

  btn.disabled = false; btn.textContent = '🚀 Generate Training Plots';
}

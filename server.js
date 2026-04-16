import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { createClient } from '@supabase/supabase-js';
import { UMAP } from 'umap-js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// ─── Supabase ───────────────────────────────────────────────────────────────
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ─── Nomic embedding ────────────────────────────────────────────────────────
async function embedText(text) {
  const res = await fetch('https://api-atlas.nomic.ai/v1/embedding/text', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.NOMIC_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'nomic-embed-text-v1.5',
      texts: [text],
      task_type: 'search_document',
      dimensionality: 768,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Nomic API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.embeddings[0];
}

// Concatenate brief fields into a single document for embedding
function briefToDocument(brief) {
  const parts = [];
  if (brief.practice_area) parts.push(`Practice: ${brief.practice_area}.`);
  if (brief.what_im_working_on) parts.push(`Working on: ${brief.what_im_working_on}.`);
  if (brief.question_im_carrying) parts.push(`Question: ${brief.question_im_carrying}.`);
  if (brief.what_push_i_want) parts.push(`Push wanted: ${brief.what_push_i_want}.`);
  return parts.join(' ');
}

// ─── UMAP projection ────────────────────────────────────────────────────────
async function reprojectAll() {
  // Fetch all briefs with embeddings
  const { data: briefs, error } = await supabase
    .from('briefs')
    .select('id, embedding')
    .not('embedding', 'is', null);

  if (error) throw error;
  if (!briefs || briefs.length === 0) return;

  const n = briefs.length;

  // Parse embedding strings → float arrays
  const embeddings = briefs.map(b => {
    if (typeof b.embedding === 'string') {
      return JSON.parse(b.embedding);
    }
    return b.embedding;
  });

  let coords;

  if (n === 1) {
    // Single brief: place at origin
    coords = [[0, 0]];
  } else if (n < 5) {
    // Too few for UMAP — spread evenly with jitter
    coords = briefs.map((_, i) => {
      const angle = (i / n) * Math.PI * 2;
      const r = 0.3 + Math.random() * 0.2;
      return [Math.cos(angle) * r, Math.sin(angle) * r];
    });
  } else {
    // Run UMAP
    const umap = new UMAP({
      nComponents: 2,
      nNeighbors: Math.min(15, n - 1),
      minDist: 0.1,
      spread: 1.0,
    });

    coords = umap.fit(embeddings);

    // Scale to [-0.8, 0.8] range (leave margin at terrain edges)
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of coords) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    coords = coords.map(([x, y]) => [
      ((x - minX) / rangeX) * 1.6 - 0.8,
      ((y - minY) / rangeY) * 1.6 - 0.8,
    ]);
  }

  // Compute Gaussian KDE density at each brief position
  const bandwidth = 0.18;
  const densities = coords.map(([px, py]) => {
    let density = 0;
    for (const [bx, by] of coords) {
      const dx = px - bx, dy = py - by;
      density += Math.exp(-(dx * dx + dy * dy) / (2 * bandwidth * bandwidth));
    }
    return density;
  });

  // Normalise densities
  const maxDensity = Math.max(...densities) || 1;

  // Write back to Supabase
  const updates = briefs.map((b, i) => ({
    id: b.id,
    x: coords[i][0],
    y: coords[i][1],
    density: densities[i] / maxDensity,
  }));

  for (const u of updates) {
    await supabase
      .from('briefs')
      .update({ x: u.x, y: u.y, density: u.density })
      .eq('id', u.id);
  }
}

// ─── API Routes ─────────────────────────────────────────────────────────────

// GET /api/briefs — fetch all briefs with coordinates and tags (no embeddings)
app.get('/api/briefs', async (req, res) => {
  try {
    const { data: briefs, error } = await supabase
      .from('briefs')
      .select('id, created_at, participant_name, practice_area, what_im_working_on, question_im_carrying, what_push_i_want, x, y, density')
      .order('created_at', { ascending: true });

    if (error) throw error;

    // Fetch tags for each brief
    const { data: briefTags, error: btError } = await supabase
      .from('brief_tags')
      .select('brief_id, tag_id, tags(label)');

    if (btError) throw btError;

    // Group tags by brief
    const tagsByBrief = {};
    for (const bt of (briefTags || [])) {
      if (!tagsByBrief[bt.brief_id]) tagsByBrief[bt.brief_id] = [];
      tagsByBrief[bt.brief_id].push(bt.tags?.label || '');
    }

    // Attach tags to briefs
    const result = (briefs || []).map(b => ({
      ...b,
      tags: tagsByBrief[b.id] || [],
    }));

    res.json(result);
  } catch (err) {
    console.error('GET /api/briefs error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/briefs — submit a new brief
app.post('/api/briefs', async (req, res) => {
  try {
    const { participant_name, practice_area, what_im_working_on, question_im_carrying, what_push_i_want, tag_ids } = req.body;

    if (!question_im_carrying) {
      return res.status(400).json({ error: 'question_im_carrying is required' });
    }

    // Build document and embed
    const doc = briefToDocument(req.body);
    const embedding = await embedText(doc);

    // Insert brief
    const { data: brief, error } = await supabase
      .from('briefs')
      .insert({
        participant_name,
        practice_area,
        what_im_working_on,
        question_im_carrying,
        what_push_i_want,
        embedding: JSON.stringify(embedding),
      })
      .select('id')
      .single();

    if (error) throw error;

    // Insert brief_tags
    if (tag_ids && tag_ids.length > 0) {
      const rows = tag_ids.map(tid => ({ brief_id: brief.id, tag_id: tid }));
      const { error: btError } = await supabase.from('brief_tags').insert(rows);
      if (btError) console.error('Tag insert error:', btError);
    }

    // Re-project all briefs via UMAP
    await reprojectAll();

    // Fetch updated brief with coordinates
    const { data: updated } = await supabase
      .from('briefs')
      .select('id, participant_name, question_im_carrying, x, y, density')
      .eq('id', brief.id)
      .single();

    res.json(updated);
  } catch (err) {
    console.error('POST /api/briefs error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/tags — list all tags
app.get('/api/tags', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('tags')
      .select('id, label, is_seed')
      .order('is_seed', { ascending: false })
      .order('label');

    if (error) throw error;
    res.json(data || []);
  } catch (err) {
    console.error('GET /api/tags error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/tags — create a new tag
app.post('/api/tags', async (req, res) => {
  try {
    const { label } = req.body;
    if (!label) return res.status(400).json({ error: 'label is required' });

    const { data, error } = await supabase
      .from('tags')
      .upsert({ label: label.toLowerCase().trim(), is_seed: false }, { onConflict: 'label' })
      .select('id, label, is_seed')
      .single();

    if (error) throw error;
    res.json(data);
  } catch (err) {
    console.error('POST /api/tags error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/edges — declare a connection
app.post('/api/edges', async (req, res) => {
  try {
    const { from_brief_id, to_brief_id, edge_type } = req.body;
    const { data, error } = await supabase
      .from('edges')
      .insert({ from_brief_id, to_brief_id, edge_type })
      .select()
      .single();

    if (error) throw error;
    res.json(data);
  } catch (err) {
    console.error('POST /api/edges error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/query — semantic search
app.post('/api/query', async (req, res) => {
  try {
    const { text, limit = 5 } = req.body;
    if (!text) return res.status(400).json({ error: 'text is required' });

    const embedding = await embedText(text);

    // Use pgvector cosine similarity via RPC or raw query
    // For now, fetch all and compute client-side (fine for <100 briefs)
    const { data: briefs, error } = await supabase
      .from('briefs')
      .select('id, participant_name, question_im_carrying, x, y, density, embedding')
      .not('embedding', 'is', null);

    if (error) throw error;

    // Cosine similarity
    function cosineSim(a, b) {
      let dot = 0, magA = 0, magB = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
      }
      return dot / (Math.sqrt(magA) * Math.sqrt(magB));
    }

    const scored = (briefs || []).map(b => {
      const emb = typeof b.embedding === 'string' ? JSON.parse(b.embedding) : b.embedding;
      return {
        id: b.id,
        participant_name: b.participant_name,
        question_im_carrying: b.question_im_carrying,
        x: b.x,
        y: b.y,
        density: b.density,
        similarity: cosineSim(embedding, emb),
      };
    });

    scored.sort((a, b) => b.similarity - a.similarity);

    res.json(scored.slice(0, limit));
  } catch (err) {
    console.error('POST /api/query error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ─── Start ──────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Civic Interplay terrain server running on http://localhost:${PORT}`);
});

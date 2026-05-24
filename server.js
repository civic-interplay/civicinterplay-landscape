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

// ─── UMAP projection (workshop-scoped) ──────────────────────────────────────
async function reprojectWorkshop(workshopId) {
  // Fetch all briefs in this workshop with embeddings
  const { data: briefs, error } = await supabase
    .from('briefs')
    .select('id, embedding')
    .eq('workshop_id', workshopId)
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

    // Scale to [-0.8, 0.8] range
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

  // Gaussian KDE density at each brief position
  const bandwidth = 0.18;
  const densities = coords.map(([px, py]) => {
    let density = 0;
    for (const [bx, by] of coords) {
      const dx = px - bx, dy = py - by;
      density += Math.exp(-(dx * dx + dy * dy) / (2 * bandwidth * bandwidth));
    }
    return density;
  });

  const maxDensity = Math.max(...densities) || 1;

  // Write back to Supabase
  for (let i = 0; i < briefs.length; i++) {
    await supabase
      .from('briefs')
      .update({
        x: coords[i][0],
        y: coords[i][1],
        density: densities[i] / maxDensity,
      })
      .eq('id', briefs[i].id);
  }
}

// ─── Workshop API Routes ────────────────────────────────────────────────────

// GET /api/workshops — list all workshops
app.get('/api/workshops', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('workshops')
      .select('id, slug, title, description, host_name, type, status, settings, created_at')
      .order('created_at', { ascending: false });
    if (error) throw error;
    res.json(data || []);
  } catch (err) {
    console.error('GET /api/workshops error:', err);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/workshops/:slug — fetch one workshop
app.get('/api/workshops/:slug', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('workshops')
      .select('id, slug, title, description, host_name, type, status, settings, created_at')
      .eq('slug', req.params.slug)
      .single();
    if (error || !data) return res.status(404).json({ error: 'workshop not found' });
    res.json(data);
  } catch (err) {
    console.error('GET /api/workshops/:slug error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/workshops — create a workshop. Add auth in production.
app.post('/api/workshops', async (req, res) => {
  try {
    const { slug, title, description, host_name, type = 'terrain', settings } = req.body;
    if (!slug || !title) return res.status(400).json({ error: 'slug and title are required' });
    const { data, error } = await supabase
      .from('workshops')
      .insert({
        slug, title, description, host_name, type,
        settings: settings || { allow_multiple_per_person: true },
      })
      .select()
      .single();
    if (error) throw error;
    res.json(data);
  } catch (err) {
    console.error('POST /api/workshops error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ─── Briefs API Routes (workshop-scoped) ────────────────────────────────────

// GET /api/briefs?workshop=slug — fetch briefs for one workshop
app.get('/api/briefs', async (req, res) => {
  try {
    const { workshop: workshopSlug } = req.query;
    if (!workshopSlug) {
      return res.status(400).json({ error: 'workshop query param is required (e.g. ?workshop=origin)' });
    }

    const { data: workshop, error: wErr } = await supabase
      .from('workshops')
      .select('id')
      .eq('slug', workshopSlug)
      .single();
    if (wErr || !workshop) {
      return res.status(404).json({ error: 'workshop not found' });
    }

    const { data: briefs, error } = await supabase
      .from('briefs')
      .select('id, created_at, participant_name, practice_area, what_im_working_on, question_im_carrying, what_push_i_want, x, y, density')
      .eq('workshop_id', workshop.id)
      .order('created_at', { ascending: true });
    if (error) throw error;

    const briefIds = (briefs || []).map(b => b.id);
    let briefTags = [];
    if (briefIds.length > 0) {
      const { data: bt, error: btError } = await supabase
        .from('brief_tags')
        .select('brief_id, tag_id, tags(label)')
        .in('brief_id', briefIds);
      if (btError) throw btError;
      briefTags = bt || [];
    }

    const tagsByBrief = {};
    for (const bt of briefTags) {
      if (!tagsByBrief[bt.brief_id]) tagsByBrief[bt.brief_id] = [];
      tagsByBrief[bt.brief_id].push(bt.tags?.label || '');
    }

    res.json((briefs || []).map(b => ({ ...b, tags: tagsByBrief[b.id] || [] })));
  } catch (err) {
    console.error('GET /api/briefs error:', err);
    res.status(500).json({ error: err.message });
  }
});

// POST /api/briefs — submit a brief to a workshop
app.post('/api/briefs', async (req, res) => {
  try {
    const {
      workshop_slug,
      participant_name,
      practice_area,
      what_im_working_on,
      question_im_carrying,
      what_push_i_want,
      tag_ids,
    } = req.body;

    if (!workshop_slug) return res.status(400).json({ error: 'workshop_slug is required' });
    if (!question_im_carrying) return res.status(400).json({ error: 'question_im_carrying is required' });

    const { data: workshop, error: wErr } = await supabase
      .from('workshops')
      .select('id, settings, status')
      .eq('slug', workshop_slug)
      .single();
    if (wErr || !workshop) return res.status(404).json({ error: 'workshop not found' });
    if (workshop.status !== 'active') {
      return res.status(403).json({ error: 'workshop is not accepting submissions' });
    }

    // Per-workshop one-per-person enforcement (off by default)
    if (workshop.settings?.allow_multiple_per_person === false && participant_name) {
      const { data: existing } = await supabase
        .from('briefs')
        .select('id')
        .eq('workshop_id', workshop.id)
        .eq('participant_name', participant_name)
        .limit(1);
      if (existing && existing.length > 0) {
        return res.status(409).json({ error: 'You have already contributed to this workshop' });
      }
    }

    const doc = briefToDocument(req.body);
    const embedding = await embedText(doc);

    const { data: brief, error } = await supabase
      .from('briefs')
      .insert({
        workshop_id: workshop.id,
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

    if (tag_ids && tag_ids.length > 0) {
      const rows = tag_ids.map(tid => ({ brief_id: brief.id, tag_id: tid }));
      const { error: btError } = await supabase.from('brief_tags').insert(rows);
      if (btError) console.error('Tag insert error:', btError);
    }

    await reprojectWorkshop(workshop.id);

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

// ─── Tags API Routes ────────────────────────────────────────────────────────

// GET /api/tags — list all tags (global, not workshop-scoped)
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

// ─── Edges API Routes ───────────────────────────────────────────────────────

// POST /api/edges — declare a connection between two briefs
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

// ─── Semantic search (workshop-scoped) ──────────────────────────────────────

// POST /api/query — semantic search within a workshop
app.post('/api/query', async (req, res) => {
  try {
    const { text, workshop_slug, limit = 5 } = req.body;
    if (!text) return res.status(400).json({ error: 'text is required' });
    if (!workshop_slug) return res.status(400).json({ error: 'workshop_slug is required' });

    const { data: workshop, error: wErr } = await supabase
      .from('workshops').select('id').eq('slug', workshop_slug).single();
    if (wErr || !workshop) return res.status(404).json({ error: 'workshop not found' });

    const embedding = await embedText(text);

    const { data: briefs, error } = await supabase
      .from('briefs')
      .select('id, participant_name, question_im_carrying, x, y, density, embedding')
      .eq('workshop_id', workshop.id)
      .not('embedding', 'is', null);
    if (error) throw error;

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
        x: b.x, y: b.y, density: b.density,
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

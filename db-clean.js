const Database = require('./node_modules/better-sqlite3');
const db = new Database('./data/agents/saphira/continuity.db');

function clean(text) {
  if (!text) return text;
  // Audio transcript
  const tidx = text.lastIndexOf('\nTranscript:\n');
  if (tidx !== -1) {
    const t = text.substring(tidx + '\nTranscript:\n'.length).trim();
    if (t.length > 0) return t;
  }
  // Timestamp marker [Mon 2026-...]
  const tsRegex = /\n\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s\d{4}-\d{2}-\d{2}\s[^\]]*\]\s*/g;
  let lastTs = null, m;
  while ((m = tsRegex.exec(text)) !== null) lastTs = m;
  if (lastTs) return text.substring(lastTs.index + lastTs[0].length).trim();
  // JSON key-value line filter
  const jsonLine = /^\s*"[a-z_]+":\s*(true|false|null|\d+|"[^"]*")\s*,?\s*$/;
  const lines = text.split('\n');
  const noisy = (l) => {
    const t = l.trim();
    return t === '' || t === '{' || t === '}' || t === '```json' || t === '```'
      || jsonLine.test(l)
      || t.startsWith('Sender (untrusted')
      || t.startsWith('Conversation info')
      || t.startsWith('[CONTINUITY')
      || t.startsWith('[STABILITY')
      || t.startsWith('You remember')
      || t.startsWith('[Audio]')
      || t.startsWith('User text:')
      || t.startsWith('[Telegram')
      || t.startsWith('[Replying');
  };
  const start = lines.findIndex(l => !noisy(l));
  if (start < 0) return '';
  return lines.slice(start).join('\n').trim();
}

const rows = db.prepare('SELECT id, user_text FROM exchanges').all();
let updated = 0;
const upd = db.prepare('UPDATE exchanges SET user_text = ? WHERE id = ?');
const txn = db.transaction(() => {
  for (const r of rows) {
    const c = clean(r.user_text || '');
    if (c !== (r.user_text || '')) { upd.run(c, r.id); updated++; }
  }
});
txn();
// Rebuild FTS
try { db.prepare("INSERT INTO fts_exchanges(fts_exchanges) VALUES('rebuild')").run(); } catch(e) {}
console.log('Cleaned', updated, 'of', rows.length, 'exchanges');
db.close();

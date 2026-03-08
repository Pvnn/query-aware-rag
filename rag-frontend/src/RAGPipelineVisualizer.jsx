import { useState, useEffect, useRef, Component } from "react";
import { motion } from "framer-motion";

// ─── ERROR BOUNDARY ───────────────────────────────────────────────────────────
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { crashed: false, msg: "" }; }
  static getDerivedStateFromError(err) { return { crashed: true, msg: String(err) }; }
  render() {
    if (this.state.crashed) {
      return (
        <div style={{ padding: 32, fontFamily: "monospace", color: "#b91c1c", background: "#fee2e2", borderRadius: 12, margin: 16 }}>
          <strong>Render error:</strong>
          <pre style={{ marginTop: 8, whiteSpace: "pre-wrap", fontSize: 12 }}>{this.state.msg}</pre>
          <button onClick={() => { this.setState({ crashed: false }); this.props.onReset?.(); }}
            style={{ marginTop: 12, padding: "8px 20px", background: "#b91c1c", color: "white", border: "none", borderRadius: 8, cursor: "pointer" }}>
            Reset
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─── MOCK DATA ────────────────────────────────────────────────────────────────
const QUERIES = [
  "How does insulin regulate blood sugar?",
  "What causes diabetes?",
  "How do SSDs work?",
  "What is machine learning?",
];

const MOCK = {
  "How does insulin regulate blood sugar?": {
    retrieved_docs: [
      { id: "D1", title: "Insulin Signaling Pathways",    snippet: "Insulin binds to surface receptors activating GLUT4 transporter translocation, facilitating cellular glucose uptake.", score: 0.94 },
      { id: "D2", title: "Pancreatic Beta Cell Function", snippet: "Beta cells in the islets of Langerhans detect blood glucose and secrete insulin proportionally to concentration.", score: 0.89 },
      { id: "D3", title: "Hepatic Glucose Metabolism",    snippet: "Insulin promotes hepatic glycogen synthesis and suppresses gluconeogenesis, reducing net glucose output.", score: 0.83 },
      { id: "D4", title: "Adipose Tissue Response",       snippet: "Adipocytes increase glucose uptake and lipogenesis while insulin suppresses lipolysis for energy storage.", score: 0.76 },
      { id: "D5", title: "Insulin Resistance Mechanisms", snippet: "Chronic hyperinsulinemia downregulates receptor expression, impairing downstream signalling and glucose uptake.", score: 0.61 },
    ],
    filtered_docs: [
      { id: "D1", kept: true }, { id: "D2", kept: true }, { id: "D3", kept: true },
      { id: "D4", kept: false, reason: "Low relevance score" },
      { id: "D5", kept: false, reason: "Below QUIT-OX threshold" },
    ],
    original_context: "Insulin binds to receptors on cell surfaces, activating intracellular signaling cascades that facilitate glucose uptake via GLUT4 transporter translocation. Beta cells in the islets of Langerhans secrete insulin in response to elevated blood glucose. In the liver, insulin promotes glycogen synthesis and inhibits gluconeogenesis.",
    compressed_context: "Insulin binds cell surface receptors activating GLUT4 translocation for glucose uptake. Beta cells secrete insulin in response to blood glucose. The liver responds with glycogen synthesis and suppressed gluconeogenesis.",
    answer: "Insulin regulates blood sugar through a coordinated multi-organ response. Pancreatic beta cells detect rising glucose and release insulin, which binds to cell surface receptors to trigger GLUT4 transporter translocation enabling peripheral cells to absorb glucose. The liver simultaneously switches from glucose production to glycogen storage.",
    metrics: { compression_ratio: 0.48, tokens_saved: 187, latency: 1.24 },
  },
  "What causes diabetes?": {
    retrieved_docs: [
      { id: "D1", title: "Type 1 Diabetes Autoimmunity",    snippet: "T1D results from autoimmune destruction of pancreatic beta cells, causing absolute insulin deficiency.", score: 0.96 },
      { id: "D2", title: "Type 2 Diabetes Pathophysiology", snippet: "T2D involves progressive insulin resistance combined with insufficient compensatory insulin secretion.", score: 0.91 },
      { id: "D3", title: "Genetic Risk Factors",            snippet: "HLA variants and TCF7L2 polymorphisms are strongly associated with T1D and T2D susceptibility.", score: 0.84 },
      { id: "D4", title: "Environmental Triggers",          snippet: "Viral infections and gut microbiome dysbiosis may trigger autoimmune beta cell destruction.", score: 0.77 },
      { id: "D5", title: "Gestational Diabetes",            snippet: "Pregnancy-induced hormonal shifts can cause transient insulin resistance in susceptible women.", score: 0.58 },
    ],
    filtered_docs: [
      { id: "D1", kept: true }, { id: "D2", kept: true }, { id: "D3", kept: true }, { id: "D4", kept: true },
      { id: "D5", kept: false, reason: "Below QUIT-OX threshold" },
    ],
    original_context: "Type 1 diabetes results from autoimmune destruction of pancreatic beta cells, leading to absolute insulin deficiency. Type 2 diabetes develops from progressive insulin resistance combined with inadequate compensatory insulin secretion, often associated with obesity.",
    compressed_context: "T1D is autoimmune beta cell destruction causing insulin deficiency. T2D involves insulin resistance and inadequate secretion linked to obesity. Genetic and environmental factors contribute.",
    answer: "Diabetes arises from two distinct mechanisms. Type 1 is an autoimmune disease where the immune system destroys insulin-producing beta cells. Type 2 develops when tissues become resistant to insulin and the pancreas can no longer compensate, often driven by obesity.",
    metrics: { compression_ratio: 0.44, tokens_saved: 142, latency: 1.08 },
  },
  "How do SSDs work?": {
    retrieved_docs: [
      { id: "D1", title: "NAND Flash Architecture",   snippet: "SSDs use floating-gate transistors that trap electrons to represent binary states without continuous power.", score: 0.95 },
      { id: "D2", title: "SSD Controller Operations", snippet: "The controller manages wear levelling, error correction (ECC), and garbage collection to maintain performance.", score: 0.88 },
      { id: "D3", title: "3D NAND Technology",        snippet: "Stacking memory cells vertically in 64-256+ layers increases density and reduces per-bit manufacturing cost.", score: 0.81 },
      { id: "D4", title: "NVMe vs SATA Interfaces",   snippet: "NVMe over PCIe enables deep command queues and high parallelism, far exceeding legacy SATA bandwidth.", score: 0.73 },
      { id: "D5", title: "SSD Market Adoption",       snippet: "SSD prices fell below $0.10/GB after 2018, accelerating displacement of HDDs in consumer storage.", score: 0.47 },
    ],
    filtered_docs: [
      { id: "D1", kept: true }, { id: "D2", kept: true }, { id: "D3", kept: true },
      { id: "D4", kept: false, reason: "Low query relevance" },
      { id: "D5", kept: false, reason: "Below QUIT-OX threshold" },
    ],
    original_context: "SSDs store data in NAND flash memory using floating-gate transistors that trap electrons to represent binary states, requiring no power to retain data. The SSD controller manages wear levelling, error correction, and garbage collection to maximise longevity.",
    compressed_context: "SSDs use NAND flash cells trapping electrons for binary storage without power. Controllers manage wear levelling, ECC, and garbage collection. 3D NAND stacks 64-256+ cell layers vertically.",
    answer: "SSDs store data in NAND flash memory using floating-gate transistors that trap electrons to represent binary 0s and 1s, retaining data without power. An onboard controller coordinates wear levelling, error correction (ECC), and garbage collection. Modern 3D NAND stacks memory layers vertically for greater density.",
    metrics: { compression_ratio: 0.47, tokens_saved: 163, latency: 1.15 },
  },
  "What is machine learning?": {
    retrieved_docs: [
      { id: "D1", title: "Supervised Learning",          snippet: "Models learn input-output mappings from labelled training examples, generalising to unseen data.", score: 0.93 },
      { id: "D2", title: "Neural Network Architectures", snippet: "Deep networks learn hierarchical representations through stacked layers trained via backpropagation.", score: 0.87 },
      { id: "D3", title: "Unsupervised Learning",        snippet: "Algorithms discover latent structure in unlabelled data via clustering and dimensionality reduction.", score: 0.80 },
      { id: "D4", title: "Reinforcement Learning",       snippet: "Agents learn policies by maximising cumulative reward through environment interaction.", score: 0.72 },
      { id: "D5", title: "ML Applications",              snippet: "ML powers image recognition, NLP, recommendation systems, fraud detection, and autonomous vehicles.", score: 0.66 },
    ],
    filtered_docs: [
      { id: "D1", kept: true }, { id: "D2", kept: true }, { id: "D3", kept: true },
      { id: "D4", kept: false, reason: "Low query relevance" },
      { id: "D5", kept: false, reason: "Below QUIT-OX threshold" },
    ],
    original_context: "Supervised learning trains models on labelled examples to generalise predictions to unseen inputs. Deep neural networks learn hierarchical feature representations through stacked layers and backpropagation. Unsupervised learning discovers hidden patterns in unlabelled data.",
    compressed_context: "ML models learn from labelled examples to generalise predictions. Deep networks build hierarchical features via backpropagation. Unsupervised learning finds patterns in unlabelled data through clustering.",
    answer: "Machine learning is a branch of AI where systems learn patterns directly from data rather than following hand-coded rules. In supervised learning, models train on labelled examples. Deep neural networks learn hierarchical representations across many stacked layers. Unsupervised methods find structure in unlabelled data.",
    metrics: { compression_ratio: 0.49, tokens_saved: 171, latency: 1.31 },
  },
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function callAPI(query) {
  try {
    const res = await fetch("/api/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
    if (!res.ok) throw new Error();
    return await res.json();
  } catch {
    await sleep(2000);
    return MOCK[query] ?? MOCK[QUERIES[0]];
  }
}

// ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
const C = {
  bg: "#f4f3ee", paper: "#fafaf7", paperAlt: "#f0efe9",
  ink: "#18181f", inkSub: "#3c3c52", inkMuted: "#8a8aa8",
  grid: "rgba(18,18,31,0.06)",
  blue: "#1d4ed8", blueL: "#dbeafe", blueB: "#93c5fd",
  green: "#047857", greenL: "#d1fae5", greenB: "#6ee7b7",
  red: "#b91c1c", redL: "#fee2e2", redB: "#fca5a5",
  amber: "#92400e", amberL: "#fef3c7", amberB: "#fcd34d",
  violet: "#5b21b6", violetL: "#ede9fe", violetB: "#c4b5fd",
  border: "rgba(18,18,31,0.11)", borderMed: "rgba(18,18,31,0.20)",
};
const PAL = {
  blue:   { a: C.blue,   l: C.blueL,   b: C.blueB   },
  violet: { a: C.violet, l: C.violetL, b: C.violetB },
  green:  { a: C.green,  l: C.greenL,  b: C.greenB  },
  amber:  { a: C.amber,  l: C.amberL,  b: C.amberB  },
  red:    { a: C.red,    l: C.redL,    b: C.redB    },
};

const STAGES = [
  { id: "query",    num: "01", label: "Query Input",           color: "blue",   icon: "search",   toggleable: false },
  { id: "retrieve", num: "02", label: "Retriever",             color: "blue",   icon: "list",     toggleable: false },
  { id: "filter",   num: "03", label: "QUIT-OX Coarse Filter", color: "violet", icon: "filter",   toggleable: true  },
  { id: "compress", num: "04", label: "EP-EXIT Compression",   color: "green",  icon: "compress", toggleable: true  },
  { id: "llm",      num: "05", label: "Reader LLM",            color: "amber",  icon: "cpu",      toggleable: false },
];

// ─── ICONS ────────────────────────────────────────────────────────────────────
function Ico({ n, sz = 18, c = "currentColor" }) {
  const s = { width: sz, height: sz, display: "block" };
  const m = {
    search:   <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>,
    list:     <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="10" x2="20" y2="10"/><line x1="4" y1="14" x2="14" y2="14"/><line x1="4" y1="18" x2="11" y2="18"/></svg>,
    filter:   <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>,
    compress: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="10" y1="14" x2="21" y2="3"/><line x1="3" y1="21" x2="14" y2="10"/></svg>,
    cpu:      <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>,
    check:    <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
    x:        <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2.2" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
    down:     <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>,
    play:     <svg style={s} viewBox="0 0 24 24" fill={c}><polygon points="5 3 19 12 5 21 5 3"/></svg>,
    chevron:  <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>,
    skip:     <svg style={s} viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12"/><line x1="5" y1="7" x2="5" y2="17" strokeDasharray="2 2"/><line x1="19" y1="7" x2="19" y2="17" strokeDasharray="2 2"/></svg>,
  };
  return m[n] || null;
}

function Spin({ sz = 14, color = C.blue }) {
  return <span style={{ display:"inline-block", width:sz, height:sz, flexShrink:0, border:"2px solid transparent", borderTopColor:color, borderRightColor:color, borderRadius:"50%", animation:"spin360 0.75s linear infinite" }} />;
}

function ScoreBar({ score }) {
  const v = Math.min(Math.max(+score || 0, 0), 1);
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, marginTop:8 }}>
      <div style={{ flex:1, height:4, background:C.border, borderRadius:99, overflow:"hidden" }}>
        <motion.div initial={{ width:0 }} animate={{ width:`${v*100}%` }} transition={{ duration:0.9, ease:[0.22,1,0.36,1] }}
          style={{ height:"100%", borderRadius:99, background: v>0.85?C.blue:v>0.7?C.inkSub:C.inkMuted }} />
      </div>
      <span style={{ fontSize:12, fontFamily:"monospace", color:C.inkMuted, minWidth:32, textAlign:"right" }}>{(v*100).toFixed(0)}%</span>
    </div>
  );
}

function Pill({ children, color="blue" }) {
  const p = PAL[color]||PAL.blue;
  return <span style={{ display:"inline-block", background:p.l, color:p.a, border:`1px solid ${p.b}`, fontSize:11, fontWeight:700, letterSpacing:"0.08em", textTransform:"uppercase", padding:"2px 8px", borderRadius:4, fontFamily:"monospace", whiteSpace:"nowrap" }}>{children}</span>;
}

function Toggle({ on, onChange, disabled }) {
  return (
    <button onClick={e=>{ e.stopPropagation(); if(!disabled) onChange(!on); }}
      style={{ display:"flex", alignItems:"center", gap:7, padding:"5px 12px 5px 7px", background:on?C.blueL:C.paperAlt, border:`1.5px solid ${on?C.blueB:C.border}`, borderRadius:20, cursor:disabled?"not-allowed":"pointer", opacity:disabled?0.5:1, transition:"all 0.2s", outline:"none" }}>
      <div style={{ width:32, height:18, borderRadius:99, position:"relative", flexShrink:0, background:on?C.blue:"#c0bfca", transition:"background 0.2s" }}>
        <div style={{ position:"absolute", top:3, left:on?15:3, width:12, height:12, borderRadius:"50%", background:"white", boxShadow:"0 1px 3px rgba(0,0,0,0.25)", transition:"left 0.2s" }} />
      </div>
      <span style={{ fontSize:11, fontWeight:700, letterSpacing:"0.09em", textTransform:"uppercase", fontFamily:"monospace", color:on?C.blue:C.inkMuted }}>{on?"Enabled":"Disabled"}</span>
    </button>
  );
}

// ─── PIPELINE NODE (left column item) ────────────────────────────────────────
function PipelineNode({ stage, status, enabled, onToggle, running }) {
  const col = PAL[stage.color]||PAL.blue;
  const isActive  = status==="active";
  const isDone    = status==="done";
  const isIdle    = status==="idle";
  const isSkipped = isDone && stage.toggleable && !enabled;
  const isOk      = isDone && !isSkipped;

  return (
    <div style={{ display:"flex", alignItems:"center", gap:12, padding:"14px 16px",minHeight:120,background:C.paper,
      border:`1.5px solid ${isActive?col.a:isOk?C.borderMed:C.border}`, borderRadius:12,
      opacity:isIdle?0.4:1, filter:isSkipped?"grayscale(0.6) opacity(0.6)":"none",
      boxShadow:isActive?`0 0 0 3px ${col.l},0 4px 20px rgba(0,0,0,0.09)`:isOk?"0 2px 8px rgba(0,0,0,0.07)":"0 1px 3px rgba(0,0,0,0.04)",
      transition:"all 0.3s ease" }}>
      {/* icon */}
      <div style={{ width:40, height:40, borderRadius:10, flexShrink:0, display:"flex", alignItems:"center", justifyContent:"center",
        background:(isOk||isActive)?col.l:C.paperAlt, border:`1.5px solid ${(isOk||isActive)?col.b:C.border}`,
        color:(isOk||isActive)?col.a:C.inkMuted, transition:"all 0.3s" }}>
        <Ico n={stage.icon} sz={18} />
      </div>
      {/* label */}
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:10, fontWeight:700, letterSpacing:"0.12em", textTransform:"uppercase", fontFamily:"monospace", color:isSkipped?C.inkMuted:col.a, marginBottom:2 }}>
          Stage {stage.num}{isSkipped?" · SKIPPED":""}
        </div>
        <div style={{ fontSize:15, fontWeight:700, color:isSkipped?C.inkMuted:C.ink, whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis", lineHeight:1.3 }}>
          {stage.label}
        </div>
      </div>
      {/* toggle + status dot */}
      <div style={{ display:"flex", alignItems:"center", gap:8, flexShrink:0 }}>
        {stage.toggleable && <Toggle on={enabled} onChange={onToggle} disabled={running} />}
        <div style={{ width:24, height:24, flexShrink:0, display:"flex", alignItems:"center", justifyContent:"center" }}>
          {isActive  && <Spin color={col.a} sz={18} />}
          {isOk      && <div style={{ width:24,height:24,borderRadius:"50%",background:C.greenL,border:`1.5px solid ${C.greenB}`,display:"flex",alignItems:"center",justifyContent:"center" }}><Ico n="check" sz={12} c={C.green}/></div>}
          {isSkipped && <div style={{ width:24,height:24,borderRadius:"50%",background:C.paperAlt,border:`1.5px dashed ${C.border}`,display:"flex",alignItems:"center",justifyContent:"center" }}><span style={{ fontSize:12,color:C.inkMuted,fontWeight:700 }}>—</span></div>}
          {isIdle    && <div style={{ width:24,height:24,borderRadius:"50%",border:`1.5px dashed ${C.border}`,background:C.paperAlt }} />}
        </div>
      </div>
    </div>
  );
}

// ─── OUTPUT PANEL (right column item) ────────────────────────────────────────
function OutputPanel({ status, enabled, stageId, children }) {
  // Query always shows
  if (stageId === "query") {
    return (
      <div style={{ background:C.paper, border:`1.5px solid ${C.borderMed}`, borderRadius:12,minHeight:120, overflow:"hidden", boxShadow:"0 2px 12px rgba(0,0,0,0.06)" }}>
        <div style={{ padding:"18px 20px" }}>{children}</div>
      </div>
    );
  }
  if (status === "idle") return (
    <div style={{ minHeight:70, background:"transparent", border:`1.5px dashed ${C.border}`, borderRadius:12, display:"flex", alignItems:"center", justifyContent:"center" }}>
      <span style={{ fontSize:13, color:C.border, fontFamily:"monospace", padding:"16px 18px" }}>Awaiting pipeline run…</span>
    </div>
  );
  if (status === "active") return (
    <div style={{ minHeight:70, background:C.paper, border:`1.5px solid ${C.border}`, borderRadius:12, display:"flex", alignItems:"center", gap:12, padding:"16px 18px" }}>
      <Spin color={C.inkMuted} sz={15}/><span style={{ fontSize:14, color:C.inkMuted, fontFamily:"monospace" }}>Processing…</span>
    </div>
  );
  // done + disabled
  if (!enabled) return (
    <div style={{ minHeight:70, background:C.paperAlt, border:`1.5px dashed ${C.border}`, borderRadius:12, display:"flex", alignItems:"center", gap:14, padding:"16px 18px", opacity:0.65 }}>
      <div style={{ width:34,height:34,borderRadius:"50%",background:C.border,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}><Ico n="skip" sz={15} c={C.inkMuted}/></div>
      <div>
        <div style={{ fontSize:13,fontWeight:700,color:C.inkMuted,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace" }}>Stage Disabled</div>
        <div style={{ fontSize:13,color:C.inkMuted,marginTop:3 }}>This stage was skipped during the pipeline run.</div>
      </div>
    </div>
  );
  // done + enabled, no children yet
  if (!children) return (
    <div style={{ minHeight:70, background:C.paper, border:`1.5px solid ${C.border}`, borderRadius:12, display:"flex", alignItems:"center", gap:12, padding:"16px 18px" }}>
      <Spin color={C.inkMuted} sz={14}/><span style={{ fontSize:14,color:C.inkMuted,fontFamily:"monospace" }}>Loading…</span>
    </div>
  );
  return (
    <div style={{ background:C.paper, border:`1.5px solid ${C.borderMed}`, borderRadius:12, overflow:"hidden", boxShadow:"0 2px 12px rgba(0,0,0,0.06)" }}>
      <div style={{ padding:"18px 20px" }}>{children}</div>
    </div>
  );
}

// ─── STAGE CONTENT ────────────────────────────────────────────────────────────
function QueryContent({ query }) {
  return (
    <div style={{ display:"flex", gap:14, flexWrap:"wrap" }}>
      <div style={{ flex:"1 1 200px", padding:"14px 17px", background:C.blueL, border:`1px solid ${C.blueB}`, borderRadius:10 }}>
        <div style={{ fontSize:11,color:C.blue,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",fontFamily:"monospace",marginBottom:8 }}>Input Query</div>
        <div style={{ fontSize:20,color:C.ink,fontWeight:600,lineHeight:1.5 }}>"{query}"</div>
      </div>
      <div style={{ flex:"0 0 auto", padding:"14px 17px", background:C.paperAlt, border:`1px solid ${C.border}`, borderRadius:10, minWidth:180 }}>
        <div style={{ fontSize:11,color:C.inkMuted,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",fontFamily:"monospace",marginBottom:8 }}>Retriever Model</div>
        <div style={{ fontSize:14,color:C.inkSub,fontFamily:"monospace",lineHeight:1.6 }}>facebook/<br/>contriever-msmarco</div>
      </div>
    </div>
  );
}

function RetrieverContent({ docs }) {
  if (!docs?.length) return null;
  return (
    <div>
      <div style={{ fontSize:13,color:C.inkMuted,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace",marginBottom:14 }}>{docs.length} Documents Retrieved — Ranked by Similarity</div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:10 }}>
        {docs.map((doc,i)=>(
          <div key={doc.id||i} style={{ padding:"13px 15px", background:C.paperAlt, border:`1px solid ${C.border}`, borderRadius:9 }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", gap:8, marginBottom:8 }}>
              <span style={{ fontSize:20,fontWeight:700,color:C.ink,lineHeight:1.3 }}>{doc.title}</span>
              <Pill color="blue">{doc.id}</Pill>
            </div>
            <p style={{ fontSize:14,color:C.inkSub,lineHeight:1.65,margin:0 }}>{doc.snippet}</p>
            <ScoreBar score={doc.score}/>
          </div>
        ))}
      </div>
    </div>
  );
}

function FilterContent({ retrievedDocs, filteredDocs }) {
  if (!retrievedDocs||!filteredDocs) return null;
  const sm = Object.fromEntries(filteredDocs.map(d=>[d.id,d]));
  const kept = filteredDocs.filter(d=>d.kept).length;
  const rem  = filteredDocs.filter(d=>!d.kept).length;
  return (
    <div>
      <div style={{ display:"flex", gap:12, marginBottom:16, flexWrap:"wrap" }}>
        {[{count:kept,label:"Kept",icon:"check",bg:C.greenL,border:C.greenB,color:C.green},{count:rem,label:"Removed",icon:"x",bg:C.redL,border:C.redB,color:C.red}].map(m=>(
          <div key={m.label} style={{ flex:"1 0 120px", padding:"12px 16px", background:m.bg, border:`1px solid ${m.border}`, borderRadius:10, display:"flex", alignItems:"center", gap:12 }}>
            <div style={{ width:32,height:32,borderRadius:"50%",background:m.color,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}><Ico n={m.icon} sz={15} c="white"/></div>
            <div>
              <div style={{ fontSize:30,fontWeight:800,color:m.color,fontFamily:"monospace",lineHeight:1 }}>{m.count}</div>
              <div style={{ fontSize:12,color:m.color,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.08em",marginTop:3 }}>{m.label}</div>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:9 }}>
        {retrievedDocs.map((doc,i)=>{
          const s=sm[doc.id]; const ok=s?s.kept:true;
          return (
            <div key={doc.id||i} style={{ padding:"12px 15px", borderRadius:9, background:ok?C.greenL:C.redL, border:`1px solid ${ok?C.greenB:C.redB}`, display:"flex", alignItems:"flex-start", gap:12 }}>
              <div style={{ width:24,height:24,borderRadius:"50%",background:ok?C.green:C.red,flexShrink:0,marginTop:2,display:"flex",alignItems:"center",justifyContent:"center" }}><Ico n={ok?"check":"x"} sz={12} c="white"/></div>
              <div style={{ flex:1,minWidth:0 }}>
                <div style={{ fontSize:20,fontWeight:700,color:ok?C.green:C.red,marginBottom:3 }}>{doc.title}</div>
                {!ok&&s?.reason&&<div style={{ fontSize:12,color:C.red,fontFamily:"monospace",opacity:0.85 }}>↳ {s.reason}</div>}
              </div>
              <Pill color={ok?"green":"red"}>{doc.id}</Pill>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CompressContent({ original, compressed, ratio }) {
  const pct = Math.round((+ratio||0)*100);
  return (
    <div>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:16, flexWrap:"wrap" }}>
        <Pill color="green">EP-EXIT Compression</Pill>
        <span style={{ fontSize:13,color:C.inkMuted,fontFamily:"monospace" }}>Evidence grouping + EXIT span filtering &nbsp;·&nbsp; <strong style={{ color:C.green }}>{pct}% reduction</strong></span>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 66px 1fr", gap:12, alignItems:"start" }}>
        <div>
          <div style={{ fontSize:11,color:C.red,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace",marginBottom:8 }}>Original Context</div>
          <div style={{ padding:"13px 15px", background:"#fff5f5", border:`1.5px solid ${C.redB}`, borderRadius:9, fontSize:14, color:C.inkSub, lineHeight:1.75, fontFamily:"monospace" }}>{original||"—"}</div>
        </div>
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", paddingTop:32 }}>
          <div style={{ width:42,height:42,borderRadius:"50%",background:C.greenL,border:`1.5px solid ${C.greenB}`,display:"flex",alignItems:"center",justifyContent:"center" }}><Ico n="down" sz={18} c={C.green}/></div>
          <div style={{ fontSize:15,fontWeight:800,color:C.green,fontFamily:"monospace",marginTop:6 }}>-{pct}%</div>
        </div>
        <div>
          <div style={{ fontSize:11,color:C.green,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace",marginBottom:8 }}>Compressed Context</div>
          <div style={{ padding:"13px 15px", background:C.greenL, border:`1.5px solid ${C.greenB}`, borderRadius:9, fontSize:14, color:C.inkSub, lineHeight:1.75, fontFamily:"monospace" }}>{compressed||"—"}</div>
        </div>
      </div>
    </div>
  );
}

function LLMContent({ context }) {
  const ctx = context||"";
  const est = Math.round(ctx.split(" ").length*1.3);
  return (
    <div>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:14, flexWrap:"wrap" }}>
        <Pill color="amber">Context Window</Pill>
        <span style={{ fontSize:13,color:C.inkMuted,fontFamily:"monospace" }}>~{est} tokens passed to reader LLM</span>
      </div>
      <div style={{ padding:"15px 17px", background:C.amberL, border:`1.5px solid ${C.amberB}`, borderRadius:10 }}>
        <div style={{ fontSize:11,color:C.amber,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace",marginBottom:8 }}>Compressed Context Input</div>
        <div style={{ fontSize:14,color:C.inkSub,lineHeight:1.75,fontFamily:"monospace" }}>{ctx||"—"}</div>
      </div>
    </div>
  );
}

function AnswerContent({ answer, metrics }) {
  const m = metrics||{};
  const pct = Math.round((+m.compression_ratio||0)*100);
  return (
    <div>
      <div style={{ padding:"18px 20px", background:C.blueL, border:`1.5px solid ${C.blueB}`, borderRadius:11, marginBottom:16 }}>
        <div style={{ fontSize:11,color:C.blue,fontWeight:700,textTransform:"uppercase",letterSpacing:"0.1em",fontFamily:"monospace",marginBottom:9 }}>Generated Answer</div>
        <div style={{ fontSize:20,color:C.ink,lineHeight:1.8 }}>{answer||"—"}</div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10 }}>
        {[{label:"Compression",value:`${pct}%`,sub:"context reduced",color:"green"},{label:"Tokens Saved",value:m.tokens_saved||0,sub:"tokens removed",color:"blue"},{label:"Latency",value:`${m.latency||0}s`,sub:"end-to-end",color:"amber"}].map(x=>{
          const p=PAL[x.color]||PAL.blue;
          return (
            <div key={x.label} style={{ padding:16, background:p.l, border:`1px solid ${p.b}`, borderRadius:10, textAlign:"center" }}>
              <div style={{ fontSize:30,fontWeight:800,color:p.a,fontFamily:"monospace",lineHeight:1 }}>{x.value}</div>
              <div style={{ fontSize:13,fontWeight:700,color:p.a,marginTop:5 }}>{x.label}</div>
              <div style={{ fontSize:12,color:C.inkMuted,marginTop:3 }}>{x.sub}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
function PipelineApp() {
  const [query,        setQuery]        = useState(QUERIES[0]);
  const [running,      setRunning]      = useState(false);
  const [phase,        setPhase]        = useState(-1);   // -1=idle, 0-4=active stage, 99=done
  const [result,       setResult]       = useState(null);
  const [dropOpen,     setDropOpen]     = useState(false);
  const [enableFilter, setEnableFilter] = useState(true);
  const [enableComp,   setEnableComp]   = useState(true);
  const dropRef  = useRef(null);
  const abortRef = useRef(false);

  useEffect(() => {
    const fn = e => { if (dropRef.current && !dropRef.current.contains(e.target)) setDropOpen(false); };
    document.addEventListener("mousedown", fn);
    return () => document.removeEventListener("mousedown", fn);
  }, []);

  function stageEnabled(id) {
    if (id === "filter")   return enableFilter;
    if (id === "compress") return enableComp;
    return true;
  }

  function statusOf(i) {
    if (phase < 0)               return "idle";
    if (running && phase === i)  return "active";
    if (phase > i)               return "done";
    return "idle";
  }

  async function handleRun() {
    if (running) return;
    abortRef.current = false;
    setRunning(true); setResult(null);

    const step = async (i, ms) => { setPhase(i); await sleep(ms); return abortRef.current; };

    if (await step(0, 420)) return;

    // Fetch data at stage 1 so result is ready before stages 2-4 go "done"
    setPhase(1);
    let data;
    try { [data] = await Promise.all([callAPI(query), sleep(900)]); }
    catch { data = MOCK[query] ?? MOCK[QUERIES[0]]; }
    if (abortRef.current) return;
    setResult(data);  // ← result set before phase advances past 1

    if (await step(2, enableFilter ? 540 : 160)) return;
    if (await step(3, enableComp  ? 540 : 160)) return;
    if (await step(4, 400)) return;

    setPhase(99);
    setRunning(false);
  }

  function handleReset() {
    abortRef.current = true;
    setResult(null); setPhase(-1); setRunning(false);
  }

  const isComplete = phase === 99 && !running && !!result;
  const llmContext = result ? (enableComp ? result.compressed_context : result.original_context) || "" : "";

  return (
    <div style={{ minHeight:"100vh", background:C.bg, fontFamily:"'Outfit','Segoe UI',sans-serif", color:C.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html,body,#root { width:100%; min-height:100vh; }
        @keyframes spin360 { to { transform:rotate(360deg); } }
        .rsp-bg { background-image:linear-gradient(${C.grid} 1px,transparent 1px),linear-gradient(90deg,${C.grid} 1px,transparent 1px); background-size:32px 32px; position:fixed; inset:0; z-index:0; pointer-events:none; }
        .outer-wrap { width:100%; max-width:1600px; margin:0 auto; padding:0 28px; }
        @media(min-width:1280px){ .outer-wrap{ padding:0 44px; } }
        @media(min-width:1500px){ .outer-wrap{ padding:0 60px; } }
        .rsp-qopt:hover { background:${C.blueL}!important; color:${C.blue}!important; }
        .rsp-run { transition:all 0.18s; cursor:pointer; }
        .rsp-run:not([disabled]):hover  { filter:brightness(1.1); transform:translateY(-1px); }
        .rsp-run:not([disabled]):active { transform:translateY(0); }
        ::-webkit-scrollbar { width:5px; }
        ::-webkit-scrollbar-thumb { background:${C.border}; border-radius:99px; }
      `}</style>

      <div className="rsp-bg" />

      {/* ── HEADER ── */}
      <header style={{ position:"sticky", top:0, zIndex:50, background:`${C.paper}f0`, backdropFilter:"blur(16px)", borderBottom:`1px solid ${C.border}` }}>
        <div className="outer-wrap">
          <div style={{ display:"flex", alignItems:"center", gap:14, height:66 }}>
            <div style={{ width:42,height:42,borderRadius:11,background:C.blue,display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,boxShadow:"0 2px 10px rgba(29,78,216,0.35)" }}>
              <Ico n="cpu" sz={22} c="white"/>
            </div>
            <div>
              <div style={{ fontSize:18,fontWeight:700,color:C.ink,letterSpacing:"-0.02em" }}>Query-Aware RAG Pipeline</div>
              <div style={{ fontSize:10,color:C.inkMuted,fontFamily:"monospace",letterSpacing:"0.08em",textTransform:"uppercase" }}>QUIT-OX · EP-EXIT · Contriever-MSMARCO · Research Visualizer</div>
            </div>
            <div style={{ marginLeft:"auto", display:"flex", alignItems:"center", gap:8, flexShrink:0, flexWrap:"wrap", justifyContent:"flex-end" }}>
              {!enableFilter && <span style={{ fontSize:11,color:C.inkMuted,fontFamily:"monospace",background:C.paperAlt,border:`1px solid ${C.border}`,padding:"4px 10px",borderRadius:20,textTransform:"uppercase",letterSpacing:"0.08em" }}>QUIT-OX off</span>}
              {!enableComp   && <span style={{ fontSize:11,color:C.inkMuted,fontFamily:"monospace",background:C.paperAlt,border:`1px solid ${C.border}`,padding:"4px 10px",borderRadius:20,textTransform:"uppercase",letterSpacing:"0.08em" }}>EP-EXIT off</span>}
              {isComplete?(
                <div style={{ display:"flex",alignItems:"center",gap:6,padding:"5px 14px",borderRadius:20,background:C.greenL,border:`1px solid ${C.greenB}` }}>
                  <div style={{ width:7,height:7,borderRadius:"50%",background:C.green }}/><span style={{ fontSize:12,color:C.green,fontWeight:700,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:"0.08em" }}>Complete</span>
                </div>
              ):running?(
                <div style={{ display:"flex",alignItems:"center",gap:6,padding:"5px 14px",borderRadius:20,background:C.amberL,border:`1px solid ${C.amberB}` }}>
                  <Spin sz={9} color={C.amber}/><span style={{ fontSize:12,color:C.amber,fontWeight:700,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:"0.08em" }}>Running</span>
                </div>
              ):(
                <div style={{ display:"flex",alignItems:"center",gap:6,padding:"5px 14px",borderRadius:20,background:C.paperAlt,border:`1px solid ${C.border}` }}>
                  <div style={{ width:7,height:7,borderRadius:"50%",background:C.border }}/><span style={{ fontSize:12,color:C.inkMuted,fontWeight:700,fontFamily:"monospace",textTransform:"uppercase",letterSpacing:"0.08em" }}>Ready</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── BODY ── */}
      <div className="outer-wrap" style={{ position:"relative", zIndex:1, paddingTop:28, paddingBottom:130 }}>

        {/*
          ┌─────────────────────────────────────────────────────────────┐
          │  TWO-COLUMN LAYOUT                                           │
          │  Left  col (fixed 380px): pipeline nodes + connectors       │
          │  Right col (flex 1):      output panels aligned per stage   │
          │                                                              │
          │  Key: both columns rendered as SEPARATE divs side by side   │
          │  inside one flex row. Each stage row in left and right       │
          │  must have the same height — achieved with minHeight on      │
          │  both the node and the output panel.                         │
          └─────────────────────────────────────────────────────────────┘
        */}

        {/* Column header labels */}
        <div style={{ display:"flex", gap:0, marginBottom:16, alignItems:"center" }}>
          {/* Left label */}
          <div style={{ width:420, flexShrink:0, display:"flex", alignItems:"center", gap:10 }}>
            <div style={{ height:1, width:14, background:C.borderMed, flexShrink:0 }}/>
            <span style={{ fontSize:11,color:C.inkMuted,fontFamily:"monospace",letterSpacing:"0.15em",textTransform:"uppercase",whiteSpace:"nowrap" }}>Pipeline Flow</span>
            <div style={{ height:1, flex:1, background:C.border }}/>
          </div>
          {/* Divider */}
          <div style={{ width:1, background:C.border, alignSelf:"stretch", margin:"0 24px", flexShrink:0 }}/>
          {/* Right label */}
          <div style={{ flex:1, display:"flex", alignItems:"center", gap:10 }}>
            <span style={{ fontSize:11,color:C.inkMuted,fontFamily:"monospace",letterSpacing:"0.15em",textTransform:"uppercase",whiteSpace:"nowrap" }}>Stage Output</span>
            <div style={{ height:1, flex:1, background:C.border }}/>
          </div>
        </div>

        {/* ── Stage rows: flex row, left col + right col ── */}
        <div style={{ display:"flex", gap:0, alignItems:"stretch" }}>

          {/* ════ LEFT COLUMN ════ */}
          <div style={{ width:380, flexShrink:0, display:"flex", flexDirection:"column" }}>
            {STAGES.map((stage, i) => {
              const st      = statusOf(i);
              const enabled = stageEnabled(stage.id);
              const lit     = phase > i;
              return (
                <div key={stage.id} style={{ display:"flex", flexDirection:"column" }}>
                  {/* Node */}
                  <PipelineNode
                    stage={stage} status={st} enabled={enabled}
                    onToggle={stage.id==="filter"?setEnableFilter:setEnableComp}
                    running={running}
                  />
                  {/* Connector arrow (not after last) */}
                  {i < STAGES.length - 1 && (
                    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", height:40, paddingLeft:20 }}>
                      <div style={{ width:2, height:26, borderRadius:1, background:lit?`linear-gradient(to bottom,${C.blue},${C.blueB})`:C.border, transition:"background 0.35s" }}/>
                      <Ico n="chevron" sz={14} c={lit?C.blue:C.border}/>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* ════ VERTICAL DIVIDER ════ */}
          <div style={{ width:1, background:C.border, margin:"0 24px", flexShrink:0 }}/>

          {/* ════ RIGHT COLUMN ════ */}
          <div style={{ flex:1, display:"flex", flexDirection:"column" }}>
            {STAGES.map((stage, i) => {
              const st      = statusOf(i);
              const enabled = stageEnabled(stage.id);

              // Build content only when data is available
              let content = null;
              try {
                if (stage.id === "query") {
                  content = <QueryContent query={query}/>;
                } else if (stage.id==="retrieve" && st==="done" && result) {
                  content = <RetrieverContent docs={result.retrieved_docs}/>;
                } else if (stage.id==="filter" && st==="done" && result && enabled) {
                  content = <FilterContent retrievedDocs={result.retrieved_docs} filteredDocs={result.filtered_docs}/>;
                } else if (stage.id==="compress" && st==="done" && result && enabled) {
                  content = <CompressContent original={result.original_context} compressed={result.compressed_context} ratio={enabled ? result.metrics?.compression_ratio : 0}/>;
                } else if (stage.id==="llm" && st==="done" && result) {
                  content = <LLMContent context={llmContext}/>;
                }
              } catch(e) {
                content = <div style={{ fontSize:13,color:C.red,fontFamily:"monospace" }}>Error: {String(e)}</div>;
              }

              // Spacer height must match node (minHeight) + connector (40px) exactly
              const isLast = i === STAGES.length - 1;
              return (
                <div key={stage.id} style={{ display:"flex", flexDirection:"column", marginBottom: isLast ? 0 : 40 }}>
                  <OutputPanel status={st} enabled={enabled} stageId={stage.id}>
                    {content}
                  </OutputPanel>
                </div>
              );
            })}
          </div>

        </div>{/* end flex row */}

        {/* ── Final Answer ── */}
        {isComplete && result && (
          <motion.div initial={{ opacity:0, y:16 }} animate={{ opacity:1, y:0 }} transition={{ duration:0.5, ease:[0.22,1,0.36,1] }} style={{ marginTop:32 }}>
            <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:14 }}>
              <div style={{ height:1, width:14, background:C.borderMed }}/>
              <span style={{ fontSize:11,color:C.inkMuted,fontFamily:"monospace",letterSpacing:"0.15em",textTransform:"uppercase",whiteSpace:"nowrap" }}>Final Answer</span>
              <div style={{ height:1, flex:1, background:C.border }}/>
            </div>
            <div style={{ background:C.paper, border:`1.5px solid ${C.borderMed}`, borderRadius:14, overflow:"hidden", boxShadow:"0 2px 14px rgba(0,0,0,0.07)" }}>
              <div style={{ padding:"22px 26px" }}><AnswerContent answer={result.answer} metrics={result.metrics}/></div>
            </div>
          </motion.div>
        )}
      </div>

      {/* ── STICKY CONTROL BAR ── */}
      <div style={{ position:"fixed", bottom:0, left:0, right:0, zIndex:60, background:`${C.paper}f4`, backdropFilter:"blur(18px)", borderTop:`1px solid ${C.border}` }}>
        <div className="outer-wrap">
          <div style={{ display:"flex", gap:12, alignItems:"center", padding:"14px 0" }}>

            {/* Query dropdown */}
            <div ref={dropRef} style={{ flex:1, position:"relative", maxWidth:620 }}>
              <button disabled={running} onClick={()=>setDropOpen(o=>!o)}
                style={{ width:"100%", padding:"12px 18px", borderRadius:10, background:C.paper, border:`1.5px solid ${C.borderMed}`, color:C.ink, fontSize:18, fontWeight:500, textAlign:"left", display:"flex", justifyContent:"space-between", alignItems:"center", cursor:running?"not-allowed":"pointer", outline:"none", fontFamily:"'Outfit','Segoe UI',sans-serif", transition:"border-color 0.2s" }}
                onMouseEnter={e=>e.currentTarget.style.borderColor=C.blue}
                onMouseLeave={e=>e.currentTarget.style.borderColor=C.borderMed}>
                <span style={{ overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", paddingRight:8 }}>{query}</span>
                <span style={{ transform:dropOpen?"rotate(180deg)":"none", transition:"transform 0.2s", flexShrink:0 }}><Ico n="chevron" sz={15} c={C.inkMuted}/></span>
              </button>
              {dropOpen && (
                <div style={{ position:"absolute", bottom:"calc(100% + 8px)", left:0, right:0, zIndex:200, background:C.paper, border:`1.5px solid ${C.borderMed}`, borderRadius:12, boxShadow:"0 8px 32px rgba(0,0,0,0.13)", overflow:"hidden" }}>
                  {QUERIES.map(q=>(
                    <button key={q} className="rsp-qopt" onClick={()=>{ setQuery(q); setDropOpen(false); handleReset(); }}
                      style={{ display:"block", width:"100%", padding:"14px 18px", cursor:"pointer", border:"none", background:q===query?C.blueL:"transparent", color:q===query?C.blue:C.inkSub, fontSize:15, textAlign:"left", fontWeight:q===query?600:400, transition:"background 0.12s,color 0.12s", fontFamily:"'Outfit','Segoe UI',sans-serif" }}>
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Run */}
            <button className="rsp-run" disabled={running} onClick={handleRun}
              style={{ padding:"12px 32px", borderRadius:10, border:"none", background:running?"#93b4f7":C.blue, color:running?"#d0dffb":"white", fontSize:15, fontWeight:700, display:"flex", alignItems:"center", gap:10, boxShadow:running?"none":"0 2px 12px rgba(29,78,216,0.32)", whiteSpace:"nowrap", minWidth:175, justifyContent:"center", fontFamily:"'Outfit','Segoe UI',sans-serif" }}>
              {running?<><Spin color="white" sz={15}/>&nbsp;Processing…</>:<><Ico n="play" sz={14} c="white"/>&nbsp;Run Pipeline</>}
            </button>

            {/* Reset */}
            {(isComplete||phase>-1) && (
              <button onClick={handleReset} style={{ background:"none", border:"none", cursor:"pointer", color:C.inkMuted, fontSize:14, textDecoration:"underline", whiteSpace:"nowrap", padding:"0 4px", fontFamily:"'Outfit','Segoe UI',sans-serif" }}>
                Reset
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function RAGPipelineVisualizer() {
  const [key, setKey] = useState(0);
  return (
    <ErrorBoundary onReset={()=>setKey(k=>k+1)}>
      <PipelineApp key={key}/>
    </ErrorBoundary>
  );
}
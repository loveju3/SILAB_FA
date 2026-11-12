const $ = (id) => document.getElementById(id);
const fields = ['device_id', 'category', 'urgency', 'duplicate_of'];
const names = {device_id:'設備', category:'分類', urgency:'急迫性', duplicate_of:'重複工單'};
const reasons = {wrong_value:'預測值與正解不同', abstention_requires_both_null:'應同時將 device 與 category 填 null 才算正確棄答', false_positive:'誤報重複：正解不是重複工單', false_negative:'漏判重複：應指向前一張工單', wrong_target:'指錯工單，同時計入 FP 與 FN', missing_prediction:'缺少這筆預測', invalid_prediction:'預測格式無效，依官方規則處理'};
const state = {data:null, page:0, devices:[], predictionFile:null};
const pageSize = 15;
function esc(value){return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function valueText(value){return value === undefined ? '未輸出' : value === null ? 'null' : typeof value === 'object' ? JSON.stringify(value) : String(value);}
function formatValue(value){return esc(valueText(value));}
function message(id, text, error=false){const el=$(id);el.textContent=text;el.hidden=!text;el.classList.toggle('error',error);}
async function api(url, options){let response;try{response=await fetch(url, options);}catch{throw new Error('無法連線到本機工具。請確認啟動工具的視窗仍開啟。');}const data=await response.json();if(!response.ok)throw new Error(data.error || '操作失敗');return data;}
function navigate(page){document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==`page-${page}`);document.querySelectorAll('.nav').forEach(el=>{el.classList.toggle('active',el.dataset.page===page);el.setAttribute('aria-current',el.dataset.page===page?'page':'false');});$('breadcrumb').textContent=page==='lookup'?'工單查詢':'評分分析';}
document.querySelectorAll('.nav').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.page)));
function setFile(file){if(!file)return;state.predictionFile=file;$('file-label').textContent=file.name;$('file-hint').textContent=`${(file.size/1024).toFixed(1)} KB · 已選取，按「開始評分」`;$('evaluation-status').hidden=true;}
$('prediction-file').addEventListener('change',e=>setFile(e.target.files[0]));
['dragenter','dragover'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.add('dragging');}));
['dragleave','drop'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');if(event==='drop')setFile(e.dataTransfer.files[0]);}));
$('gold-source').addEventListener('change',()=>{$('gold-file-label').hidden=$('gold-source').value!=='custom';});
$('gold-file').addEventListener('change',()=>{$('gold-file-label').textContent=$('gold-file').files[0]?.name||'選擇標準答案檔';});
async function evaluate(baseline=false){
  const controls=[$('evaluate'),$('baseline'),$('prediction-file'),$('gold-source'),$('gold-file')];
  controls.forEach(el=>el.disabled=true);
  message('evaluation-status',baseline?'正在執行 Baseline 並評分…':'正在計算分數與整理錯誤…');
  try{
    const gold=$('gold-source').value;
    const payload={gold,baseline};
    if(gold==='custom'){
      const file=$('gold-file').files[0];if(!file)throw new Error('請先選擇自訂標準答案檔。');
      if(file.size>8*1024*1024)throw new Error('答案檔請小於 8 MB。');
      payload.gold_content=await file.text();payload.gold_name=file.name;
    }
    if(!baseline){
      const file=state.predictionFile;if(!file)throw new Error('請先點擊上方區域選擇預測 JSONL 檔，或載入 Baseline 範例。');
      if(file.size>8*1024*1024)throw new Error('預測檔請小於 8 MB。');
      payload.prediction_content=await file.text();payload.prediction_name=file.name;
    }
    const result=await api('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    state.data=result;state.page=0;$('error-search').value='';$('field-filter').value='';renderResults();
    message('evaluation-status',`評分完成 · ${result.summary.gold_count} 筆工單 · 報告已儲存在 helper_tool/reports/ui/${result.run_id}`);
  }catch(error){message('evaluation-status',error.message,true);}
  finally{controls.forEach(el=>el.disabled=false);}
}
$('evaluate').addEventListener('click',()=>evaluate());$('baseline').addEventListener('click',()=>evaluate(true));
function renderResults(){
  const {summary:s,label,downloads}=state.data;
  $('empty-evaluation').hidden=true;$('results').hidden=false;$('run-label').textContent=label;
  $('download-csv').href=downloads['errors.csv'];$('download-json').href=downloads['summary.json'];$('download-errors').href=downloads['errors.jsonl'];
  const cards=[['總分',s.total,'加權結果 · 0～1 分',true],['category',s.metrics.category,'macro-F1 · 權重 0.4'],['device',s.metrics.device,'accuracy · 權重 0.2'],['urgency',s.metrics.urgency,'macro-F1 · 權重 0.2'],['duplicate',s.metrics.duplicate,'ID F1 · 權重 0.2']];
  $('metrics').innerHTML=cards.map(([name,score,desc,total])=>`<div class="metric-card ${total?'total':''}"><div class="metric-name">${name}<span>${total?'SCORE':''}</span></div><div class="metric-value">${total?(score*100).toFixed(2):score.toFixed(3)}${total?'<small> / 100</small>':''}</div><div class="metric-desc">${desc}${total?` · ${score.toFixed(6)}`:''}</div><div class="metric-track"><span style="width:${Math.max(0,Math.min(100,score*100))}%"></span></div></div>`).join('');
  const warnings=[];
  if(s.missing_ids.length)warnings.push(`缺少 ${s.missing_ids.length} 筆預測（${s.missing_ids.slice(0,5).join('、')}${s.missing_ids.length>5?'…':''}）`);
  if(s.extra_ids.length)warnings.push(`多餘 ${s.extra_ids.length} 個 ID 不計分（${s.extra_ids.slice(0,5).join('、')}${s.extra_ids.length>5?'…':''}）`);
  if(Object.keys(s.invalid_predictions).length)warnings.push(`${Object.keys(s.invalid_predictions).length} 筆無效預測`);
  if(s.suggestion_warning_ids.length)warnings.push(`${s.suggestion_warning_ids.length} 筆缺少處置建議（不計分）`);
  $('validation').hidden=!warnings.length;$('validation').textContent=warnings.join('；');
  const max=Math.max(...Object.values(s.wrong_field_counts),1);
  $('error-bars').innerHTML=fields.map(field=>`<button class="bar-row" data-filter="${field}" aria-label="篩選 ${names[field]} 錯題"><span class="bar-label">${field}</span><span class="bar-track"><span style="width:${s.wrong_field_counts[field]/max*100}%"></span></span><span class="bar-count">${s.wrong_field_counts[field]}</span></button>`).join('');
  document.querySelectorAll('[data-filter]').forEach(el=>el.addEventListener('click',()=>{$('field-filter').value=el.dataset.filter;state.page=0;renderErrors();$('field-filter').scrollIntoView({behavior:'smooth',block:'center'});}));
  renderMatrix();renderErrors();
}
function renderMatrix(){
  const classes=Object.keys(state.data.summary.per_class.category), columns=[...classes,'未輸出'];
  const matrix=classes.map(()=>columns.map(()=>0));
  classes.forEach((c,i)=>{matrix[i][i]=state.data.summary.per_class.category[c].support;});
  state.data.errors.forEach(e=>{if(!e.differences.category || !classes.includes(e.gold.category))return;const r=classes.indexOf(e.gold.category);let c=classes.indexOf(e.differences.category.predicted);if(c<0)c=classes.length;matrix[r][r]--;matrix[r][c]++;});
  const max=Math.max(...matrix.flat(),1);
  $('matrix').innerHTML=`<table class="matrix"><thead><tr><th></th>${columns.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${classes.map((c,r)=>`<tr><th>${c}</th>${columns.map((_,i)=>`<td class="${r===i?'diagonal':'off'}" style="background:${r===i?`rgba(105,164,117,${.07+matrix[r][i]/max*.5})`:`rgba(210,155,100,${.04+matrix[r][i]/max*.4})`}" title="正解 ${c} → 預測 ${columns[i]}：${matrix[r][i]} 筆">${matrix[r][i]||'·'}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderErrors(){
  if(!state.data)return;
  const query=$('error-search').value.trim().toLowerCase(),field=$('field-filter').value;
  const filtered=state.data.errors.filter(e=>(!field||e.wrong_fields.includes(field))&&(!query||`${e.ticket_id} ${e.raw_text} ${e.reporter}`.toLowerCase().includes(query)));
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize));state.page=Math.min(state.page,pages-1);
  $('error-count').textContent=`${state.data.errors.length} / ${state.data.summary.gold_count} 筆有錯`;
  const rows=filtered.slice(state.page*pageSize,(state.page+1)*pageSize);
  $('error-rows').innerHTML=rows.length?rows.map(e=>`<tr><td class="ticket-id">${esc(e.ticket_id)}</td><td><div class="raw-preview">${esc((e.raw_text||'').slice(0,90))}${(e.raw_text||'').length>90?'…':''}</div></td><td><div class="pills">${e.wrong_fields.map(f=>`<span class="pill ${f}">${names[f]}</span>`).join('')}</div></td><td><button class="detail-button" data-detail="${esc(e.ticket_id)}">查看差異 ↗</button></td></tr>`).join(''):'<tr><td colspan="4" class="muted">沒有符合條件的錯題。</td></tr>';
  $('page-info').textContent=`${filtered.length} 筆符合 · 第 ${state.page+1} / ${pages} 頁`;$('prev').disabled=state.page===0;$('next').disabled=state.page>=pages-1;
  document.querySelectorAll('[data-detail]').forEach(el=>el.addEventListener('click',()=>showDetail(el.dataset.detail)));
}
['error-search','field-filter'].forEach(id=>$(id).addEventListener(id==='error-search'?'input':'change',()=>{state.page=0;renderErrors();}));
$('prev').addEventListener('click',()=>{state.page--;renderErrors();});$('next').addEventListener('click',()=>{state.page++;renderErrors();});
function showDetail(id){
  const e=state.data.errors.find(error=>error.ticket_id===id);if(!e)return;
  $('detail-title').textContent=e.ticket_id;
  $('detail-content').innerHTML=`<p class="provenance">${esc(e.source_file)} : ${e.source_line}</p><div class="ticket-meta"><span>${esc(e.reporter)}</span><span>${esc(e.created_at)}</span></div><div class="raw-text">${esc(e.raw_text)}</div>${e.validation_errors.length?`<div class="message error">${e.validation_errors.map(esc).join('<br>')}</div>`:''}<table class="compare-table"><thead><tr><th>欄位</th><th>標準答案</th><th>本次預測</th><th>判讀</th></tr></thead><tbody>${fields.map(f=>`<tr class="${e.differences[f]?'wrong':''}"><td>${esc(f)}</td><td>${linkedValue(f,e.gold[f])}</td><td>${e.prediction?linkedValue(f,e.prediction[f]):'缺少整筆預測'}</td><td class="reason">${e.differences[f]?esc(reasons[e.differences[f].reason]||e.differences[f].reason):f==='category'&&e.gold[f]===null?'不納入分類 F1':'一致'}</td></tr>`).join('')}</tbody></table><p class="detail-note">category 的標準答案為 null 時不計入分類 F1；設備棄答需 device 與 category 同時為 null。點擊重複工單編號可查原文。</p>`;
  bindTicketLinks($('detail-content'));$('detail-dialog').showModal();
}
function linkedValue(field,value){return field==='duplicate_of'&&typeof value==='string'&&value?`<button class="link-button" data-ticket="${esc(value)}">${esc(value)}</button>`:formatValue(value);}
$('close-detail').addEventListener('click',()=>$('detail-dialog').close());$('detail-dialog').addEventListener('click',e=>{if(e.target===$('detail-dialog')){const rect=e.target.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)e.target.close();}});
function deviceName(id){return state.devices.find(d=>d.device_id===id)?.name;}
function bindTicketLinks(container){container.querySelectorAll('[data-ticket]').forEach(el=>el.addEventListener('click',()=>{if($('detail-dialog').open)$('detail-dialog').close();navigate('lookup');$('ticket-query').value=el.dataset.ticket;lookup();}));}
function ticketCard(match){const r=match.record;const source=match.source_file.split('/').pop();return `<article class="panel ticket-card"><div class="ticket-card-head"><h2>${esc(r.ticket_id)}</h2><span class="tag">${esc(source)}</span></div><p class="provenance">${esc(match.source_file)} : ${match.source_line}</p><div class="ticket-meta"><span>回報人 ${esc(r.reporter)}</span><span>${esc(r.created_at)}</span>${r.status?`<span>狀態 ${esc(r.status)}</span>`:''}</div><div class="raw-text">${esc(r.raw_text)}</div>${fields.some(f=>f in r)?`<div class="label-list">${fields.filter(f=>f in r).map(f=>`<div class="label-item"><small>${f}</small>${linkedValue(f,r[f])}${f==='device_id'&&deviceName(r[f])?` · ${esc(deviceName(r[f]))}`:''}</div>`).join('')}</div>`:'<p class="detail-note">此資料來源未提供四欄標準答案。</p>'}<details class="raw-json"><summary>查看完整 JSON</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></article>`;}
async function lookup(){
  const ids=$('ticket-query').value.trim().split(/[\s,，]+/).filter(Boolean).join(' ');if(!ids)return;
  $('lookup-button').disabled=true;message('lookup-status','正在查詢工單…');
  try{const data=await api(`/api/lookup?id=${encodeURIComponent(ids)}`);$('lookup-results').innerHTML=Object.entries(data.results).map(([id,matches])=>matches.length?matches.map(ticketCard).join(''):`<article class="panel no-match"><h2>${esc(id)} · 查無資料</h2><p>請確認工單編號。查詢採精確比對，包含大小寫與前置零。</p></article>`).join('');bindTicketLinks($('lookup-results'));const count=Object.values(data.results).reduce((sum,m)=>sum+m.length,0);message('lookup-status',`查詢完成 · 找到 ${count} 筆資料`);}
  catch(error){message('lookup-status',error.message,true);}finally{$('lookup-button').disabled=false;}
}
$('lookup-form').addEventListener('submit',e=>{e.preventDefault();lookup();});bindTicketLinks(document.querySelector('.quick-queries'));
api('/api/info').then(info=>{state.devices=info.devices;$('source-counts').innerHTML=info.sources.map(s=>`<span>${esc(s.name)} <b>${s.count.toLocaleString()}</b> 筆</span>`).join('');}).catch(error=>message('lookup-status',error.message,true));

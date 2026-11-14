const $ = (id) => document.getElementById(id);
const fields = ['device_id', 'category', 'urgency', 'duplicate_of'];
const names = {device_id:'設備', category:'分類', urgency:'急迫性', duplicate_of:'重複工單'};
const reasons = {wrong_value:'預測值與正解不同', abstention_requires_both_null:'應同時將 device 與 category 填 null 才算正確棄答', false_positive:'誤報重複：正解不是重複工單', false_negative:'漏判重複：應指向前一張工單', wrong_target:'指錯工單，同時計入 FP 與 FN', missing_prediction:'缺少這筆預測', invalid_prediction:'預測格式無效，依官方規則處理'};
const state = {data:null, page:0, devices:[], predictionFile:null};
const pageSize = 15;
function esc(value){return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function valueText(value){return value === undefined ? '未輸出' : value === null ? 'null' : typeof value === 'object' ? JSON.stringify(value) : String(value);}
function formatValue(value){return esc(valueText(value));}
function scoreText(summary){return `加權總分 ${summary.total.toFixed(6)} · 分類 macro-F1 ${summary.metrics.category.toFixed(6)}`;}
function message(id, text, error=false){const el=$(id);el.textContent=text;el.hidden=!text;el.classList.toggle('error',error);}
async function api(url, options){let response;try{response=await fetch(url, options);}catch{throw new Error('無法連線到本機工具。請確認啟動工具的視窗仍開啟。');}const data=await response.json();if(!response.ok)throw new Error(data.error || '操作失敗');return data;}
function navigate(page){document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==`page-${page}`);document.querySelectorAll('.nav').forEach(el=>{el.classList.toggle('active',el.dataset.page===page);el.setAttribute('aria-current',el.dataset.page===page?'page':'false');});$('breadcrumb').textContent=({lookup:'工單查詢',agent:'智慧問答 Agent',training:'模型訓練'})[page]||'評分分析';history.replaceState(null,'',`#${page}`);if(page==='agent')refreshAgent();else clearTimeout(agentTimer);if(page==='training')refreshTraining();else clearTimeout(trainingTimer);}
document.querySelectorAll('.nav').forEach(el=>el.addEventListener('click',()=>navigate(el.dataset.page)));
function setFile(file){if(!file)return;state.predictionFile=file;$('file-label').textContent=file.name;$('file-hint').textContent=`${(file.size/1024).toFixed(1)} KB · 已選取，按「開始評分」`;$('evaluation-status').hidden=true;}
$('prediction-file').addEventListener('change',e=>setFile(e.target.files[0]));
['dragenter','dragover'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.add('dragging');}));
['dragleave','drop'].forEach(event=>$('drop-zone').addEventListener(event,e=>{e.preventDefault();$('drop-zone').classList.remove('dragging');if(event==='drop')setFile(e.dataTransfer.files[0]);}));
$('gold-source').addEventListener('change',()=>{$('gold-file-label').hidden=$('gold-source').value!=='custom';});
$('gold-file').addEventListener('change',()=>{$('gold-file-label').textContent=$('gold-file').files[0]?.name||'選擇標準答案檔';});
async function evaluate(baseline=false){
  const controls=document.querySelectorAll('.setup input,.setup select,.setup button,#baseline');
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
    acceptResult(result);
    message('evaluation-status',`評分完成 · ${result.summary.gold_count} 筆工單 · ${scoreText(result.summary)} · 總耗時 ${result.timing.total_seconds.toFixed(3)} 秒 · 報告已儲存在 helper_tool/reports/ui/${result.run_id}`);
  }catch(error){message('evaluation-status',error.message,true);}
  finally{controls.forEach(el=>el.disabled=false);}
}
$('evaluate').addEventListener('click',()=>evaluate());$('baseline').addEventListener('click',()=>evaluate(true));
function acceptResult(result){
  state.data=result;state.page=0;$('error-search').value='';$('field-filter').value='';
  $('result-filter').value=result.summary?'errors':'all';
  $('result-filter').querySelector('[value="all"]').disabled=!result.records;
  $('result-filter').querySelector('[value="errors"]').disabled=!result.summary;
  $('field-filter').disabled=!result.summary;
  renderResults();
}
$('test-dataset').addEventListener('change',()=>{$('test-file-label').hidden=$('test-dataset').value!=='custom';});
$('test-file').addEventListener('change',()=>{$('test-file-label').textContent=$('test-file').files[0]?.name||'選擇測試資料檔';});
$('test-limit').addEventListener('change',()=>{$('limit-hint').hidden=$('test-limit').value!=='2gb';});
$('experiment-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const controls=document.querySelectorAll('.setup input,.setup select,.setup button,#baseline');
  controls.forEach(el=>el.disabled=true);
  message('evaluation-status','正在執行預測、評分與整理結果，請稍候…');
  try{
    const payload={dataset:$('test-dataset').value,method:$('test-method').value,limit:$('test-limit').value,output_dir:$('output-dir').value,model_path:$('model-path').value};
    if(payload.dataset==='custom'){
      const file=$('test-file').files[0];if(!file)throw new Error('請選擇測試資料檔。');
      if(file.size>8*1024*1024)throw new Error('測試資料檔請小於 8 MB。');
      payload.dataset_content=await file.text();payload.dataset_name=file.name;
    }
    const result=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    acceptResult(result);
    const report=result.summary?`${scoreText(result.summary)} · ${result.errors.length} 筆有錯`:'沒有標準答案，已產生預測但未評分';
    message('evaluation-status',`完成 ${result.execution.count} 筆 · 總耗時 ${result.timing.total_seconds.toFixed(3)} 秒 · ${report}\n預測檔：${result.execution.output_path}`);
    $('results').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){message('evaluation-status',error.message,true);}
  finally{controls.forEach(el=>el.disabled=false);}
});
function renderResults(){
  const {summary:s,label,downloads}=state.data;
  $('empty-evaluation').hidden=true;$('results').hidden=false;$('run-label').textContent=label;
  for(const [id,name] of [['download-csv','errors.csv'],['download-json','summary.json'],['download-errors','errors.jsonl'],['download-predictions','predictions.jsonl']]){
    $(id).hidden=!downloads[name];if(downloads[name])$(id).href=downloads[name];
  }
  const execution=state.data.execution;
  const timing=state.data.timing;
  $('run-timing').hidden=!timing;
  if(timing){
    $('total-time').textContent=`${timing.total_seconds.toFixed(3)} 秒`;
    $('total-time-note').textContent=timing.prediction_seconds===null?'讀取預測、評分與儲存報告':s?'準備資料、預測、評分與儲存報告':'準備資料、預測與儲存結果';
    $('prediction-time').textContent=timing.prediction_seconds===null?'未執行預測':`${timing.prediction_seconds.toFixed(3)} 秒`;
    $('prediction-time-note').textContent=timing.prediction_seconds===null?'本次只評分上傳的預測檔':execution?.limit==='2gb'?'包含容器啟動、預測與容器清理':execution?'包含啟動、載入與預測輸出':'執行本機 Baseline 函式';
  }
  $('output-details').hidden=!execution;
  if(execution)$('output-details').textContent=`儲存位置：${execution.output_path} · ${execution.limit==='2gb'?'2 GB / 禁止 swap':'本機未限制記憶體'}`;
  $('score-charts').hidden=!s;
  if(!s){
    $('metrics').innerHTML=`<div class="metric-card"><div class="metric-name">預測完成</div><div class="metric-value">${execution.count}<small> 筆</small></div><div class="metric-desc">無標準答案，未計算得分</div></div>`;
    $('validation').hidden=true;renderErrors();return;
  }
  const cards=[['加權總分',s.total,'四欄加權 · 0～1 分',true],['分類 macro-F1',s.metrics.category,'五類平均 F1 · 權重 0.4',false,true],['device',s.metrics.device,'accuracy · 權重 0.2'],['urgency',s.metrics.urgency,'macro-F1 · 權重 0.2'],['duplicate',s.metrics.duplicate,'ID F1 · 權重 0.2']];
  $('metrics').innerHTML=cards.map(([name,score,desc,total,precise])=>`<div class="metric-card ${total?'total':''}"><div class="metric-name">${name}<span>${total?'SCORE':''}</span></div><div class="metric-value ${precise?'precise':''}">${total?(score*100).toFixed(2):score.toFixed(precise?6:3)}${total?'<small> / 100</small>':''}</div><div class="metric-desc">${desc}${total?` · ${score.toFixed(6)}`:''}</div><div class="metric-track"><span style="width:${Math.max(0,Math.min(100,score*100))}%"></span></div></div>`).join('');
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
  const source=$('result-filter').value==='all'?(state.data.records||state.data.errors):state.data.errors;
  const filtered=source.filter(e=>(!field||e.wrong_fields.includes(field))&&(!query||`${e.ticket_id} ${e.raw_text} ${e.reporter}`.toLowerCase().includes(query)));
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize));state.page=Math.min(state.page,pages-1);
  $('error-count').textContent=state.data.summary?`${state.data.errors.length} / ${state.data.summary.gold_count} 筆有錯`:`${state.data.execution.count} 筆 · 未評分`;
  const rows=filtered.slice(state.page*pageSize,(state.page+1)*pageSize);
  $('error-rows').innerHTML=rows.length?rows.map(e=>`<tr><td class="ticket-id">${esc(e.ticket_id)}</td><td><div class="raw-preview">${esc((e.raw_text||'').slice(0,90))}${(e.raw_text||'').length>90?'…':''}</div></td><td><div class="pills">${e.wrong_fields.length?e.wrong_fields.map(f=>`<span class="pill ${f}">${names[f]}</span>`).join(''):`<span class="pill correct">${state.data.summary?'一致':'未評分'}</span>`}</div></td><td><button class="detail-button" data-detail="${esc(e.ticket_id)}">查看差異 ↗</button></td></tr>`).join(''):'<tr><td colspan="4" class="muted">沒有符合條件的判斷結果。</td></tr>';
  $('page-info').textContent=`${filtered.length} 筆符合 · 第 ${state.page+1} / ${pages} 頁`;$('prev').disabled=state.page===0;$('next').disabled=state.page>=pages-1;
  document.querySelectorAll('[data-detail]').forEach(el=>el.addEventListener('click',()=>showDetail(el.dataset.detail)));
}
['error-search','field-filter','result-filter'].forEach(id=>$(id).addEventListener(id==='error-search'?'input':'change',()=>{state.page=0;renderErrors();}));
$('prev').addEventListener('click',()=>{state.page--;renderErrors();});$('next').addEventListener('click',()=>{state.page++;renderErrors();});
function showDetail(id){
  const e=(state.data.records||state.data.errors).find(error=>error.ticket_id===id);if(!e)return;
  $('detail-title').textContent=e.ticket_id;
  $('detail-content').innerHTML=`<p class="provenance">${esc(e.source_file)} : ${e.source_line}</p><div class="ticket-meta"><span>${esc(e.reporter)}</span><span>${esc(e.created_at)}</span></div><div class="raw-text">${esc(e.raw_text)}</div>${e.validation_errors.length?`<div class="message error">${e.validation_errors.map(esc).join('<br>')}</div>`:''}<table class="compare-table"><thead><tr><th>欄位</th><th>標準答案</th><th>本次預測</th><th>判讀</th></tr></thead><tbody>${fields.map(f=>`<tr class="${e.differences[f]?'wrong':''}"><td>${esc(f)}</td><td>${e.gold?linkedValue(f,e.gold[f]):'未提供'}</td><td>${e.prediction?linkedValue(f,e.prediction[f]):'缺少整筆預測'}</td><td class="reason">${e.differences[f]?esc(reasons[e.differences[f].reason]||e.differences[f].reason):!e.gold?'未評分':f==='category'&&e.gold[f]===null?'不納入分類 F1':'一致'}</td></tr>`).join('')}</tbody></table><p class="detail-note">處置建議：${esc(e.prediction?.suggestion||'未提供')}</p><p class="detail-note">category 的標準答案為 null 時不計入分類 F1；設備棄答需 device 與 category 同時為 null。點擊重複工單編號可查原文。</p>`;
  bindTicketLinks($('detail-content'));$('detail-dialog').showModal();
}
function linkedValue(field,value){return field==='duplicate_of'&&typeof value==='string'&&value?`<button class="link-button" data-ticket="${esc(value)}">${esc(value)}</button>`:formatValue(value);}
$('close-detail').addEventListener('click',()=>$('detail-dialog').close());$('detail-dialog').addEventListener('click',e=>{if(e.target===$('detail-dialog')){const rect=e.target.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)e.target.close();}});
function deviceName(id){return state.devices.find(d=>d.device_id===id)?.name;}
function bindTicketLinks(container){container.querySelectorAll('[data-ticket]').forEach(el=>el.addEventListener('click',()=>{if(state.data?.records?.some(r=>r.ticket_id===el.dataset.ticket)){showDetail(el.dataset.ticket);return;}if($('detail-dialog').open)$('detail-dialog').close();navigate('lookup');$('ticket-query').value=el.dataset.ticket;lookup();}));}
function ticketCard(match){const r=match.record;const source=match.source_file.split('/').pop();return `<article class="panel ticket-card"><div class="ticket-card-head"><h2>${esc(r.ticket_id)}</h2><span class="tag">${esc(source)}</span></div><p class="provenance">${esc(match.source_file)} : ${match.source_line}</p><div class="ticket-meta"><span>回報人 ${esc(r.reporter)}</span><span>${esc(r.created_at)}</span>${r.status?`<span>狀態 ${esc(r.status)}</span>`:''}</div><div class="raw-text">${esc(r.raw_text)}</div>${fields.some(f=>f in r)?`<div class="label-list">${fields.filter(f=>f in r).map(f=>`<div class="label-item"><small>${f}</small>${linkedValue(f,r[f])}${f==='device_id'&&deviceName(r[f])?` · ${esc(deviceName(r[f]))}`:''}</div>`).join('')}</div>`:'<p class="detail-note">此資料來源未提供四欄標準答案。</p>'}<details class="raw-json"><summary>查看完整 JSON</summary><pre>${esc(JSON.stringify(r,null,2))}</pre></details></article>`;}
async function lookup(){
  const ids=$('ticket-query').value.trim().split(/[\s,，]+/).filter(Boolean).join(' ');if(!ids)return;
  $('lookup-button').disabled=true;message('lookup-status','正在查詢工單…');
  try{const data=await api(`/api/lookup?id=${encodeURIComponent(ids)}`);$('lookup-results').innerHTML=Object.entries(data.results).map(([id,matches])=>matches.length?matches.map(ticketCard).join(''):`<article class="panel no-match"><h2>${esc(id)} · 查無資料</h2><p>請確認工單編號。查詢採精確比對，包含大小寫與前置零。</p></article>`).join('');bindTicketLinks($('lookup-results'));const count=Object.values(data.results).reduce((sum,m)=>sum+m.length,0);message('lookup-status',`查詢完成 · 找到 ${count} 筆資料`);}
  catch(error){message('lookup-status',error.message,true);}finally{$('lookup-button').disabled=false;}
}
$('lookup-form').addEventListener('submit',e=>{e.preventDefault();lookup();});bindTicketLinks(document.querySelector('.quick-queries'));
api('/api/info').then(info=>{state.devices=info.devices;for(const s of info.sources){for(const id of ['test-dataset','gold-source']){const option=$(id).querySelector(`option[value="${s.name}"]`);if(option)option.disabled=s.available===false;}}if(info.default_output_dir)$('output-dir').value=info.default_output_dir;$('source-counts').innerHTML=info.sources.map(s=>`<span>${esc(s.name)} <b>${s.available===false?'未提供':s.count===null?'僅供驗證':s.count.toLocaleString()+' 筆'}</b></span>`).join('');}).catch(error=>message('lookup-status',error.message,true));

// Training is a persisted backend job; polling refreshes the training page.
let trainingTimer=null, trainingJobs=[], trainingStarting=false;
const jobStates={queued:'等待中',running:'訓練中',succeeded:'完成',failed:'失敗',interrupted:'已中斷'};
const trainingNames={'bge-finetune':'BGE 聯合微調比較','bge-frozen':'凍結 BGE 四候選',tfidf:'TF-IDF 四候選'};
function metricNumber(value){return typeof value==='number'?value.toFixed(6):'—';}
function renderTraining(job){
  const epochs=job.events.filter(e=>e.event==='epoch');
  const results=job.comparison?.results||job.events.filter(e=>e.event==='baseline');
  const active=['queued','running'].includes(job.state);
  $('start-training').disabled=trainingStarting||trainingJobs.some(j=>['queued','running'].includes(j.state));
  const progress=job.progress||{percent:job.state==='succeeded'?100:0,determinate:job.state==='succeeded'};
  if(progress.determinate||!active)$('training-progress').value=progress.percent;else $('training-progress').removeAttribute('value');
  $('training-progress-value').textContent=`${jobStates[job.state]}${progress.determinate||!active?' · '+Math.floor(progress.percent)+'%':''}`;
  $('training-progress-label').textContent=progress.label||job.message;
  const models=job.state==='succeeded'?(results.length?results.map(r=>({name:r.variant,path:r.model_path,method:r.method||(['bge-lr','bge-svc'].includes(r.variant)?r.variant:'bge-finetuned')})):(job.config.kind==='tfidf'?['tfidf-lr','tfidf-svc']:['bge-lr','bge-svc']).map(method=>({name:method,method,path:`${job.directory}/models/${method}.joblib`}))):[];
  $('training-results').innerHTML=`<p><strong>${esc(jobStates[job.state]||job.state)}</strong> · ${esc(job.message)}${typeof job.elapsed_seconds==='number'?` · ${job.elapsed_seconds.toFixed(1)} 秒`:''}</p><p class="provenance">${esc(job.directory)}<br>來源：${esc(job.config.source_name)} · ${job.config.count} 筆 · ${esc(trainingNames[job.config.kind])} · ${esc(job.config.backend||'cpu')}</p>${results.length?`<h4>相同 history 驗證段的比較</h4><div class="table-scroll"><table class="compare-table"><thead><tr><th>方法</th><th>文字分類 F1</th><th>六類 F1（選型）</th><th>null 召回</th><th>四欄分類 F1</th><th>設備正確率</th><th>急迫性 F1</th><th>重複 F1</th><th>四欄總分</th><th>最佳 epoch</th></tr></thead><tbody>${results.map(r=>`<tr><td>${esc(r.variant)}</td><td>${metricNumber(r.validation.category_macro_f1)}</td><td>${metricNumber(r.validation.six_class_macro_f1)}</td><td>${metricNumber(r.validation.unknown_recall)}</td><td>${metricNumber(r.pipeline_metrics?.category)}</td><td>${metricNumber(r.pipeline_metrics?.device)}</td><td>${metricNumber(r.pipeline_metrics?.urgency)}</td><td>${metricNumber(r.pipeline_metrics?.duplicate)}</td><td>${metricNumber(r.pipeline_total)}</td><td>${r.best_epoch||'—'}</td></tr>`).join('')}</tbody></table></div><p class="detail-note">${job.comparison?.refit_full_history?'表中為 train-only 選型驗證成績；保存模型已用完整 history 重訓，權重不同，不可把 history 回測當成獨立測試。':'驗證段同時用於選 epoch，因此是選型驗證成績，不是獨立測試；保存最佳 train-only checkpoint，不以完整 history 重訓。'} 未自動替換預設模型。${esc(job.comparison?.provenance||'')}</p>`:''}${epochs.length?`<h4>逐 epoch 訓練／驗證</h4><div class="table-scroll"><table class="compare-table"><thead><tr><th>方法／epoch</th><th>train loss</th><th>訓練六類 F1</th><th>驗證六類 F1</th><th>驗證分類 F1</th></tr></thead><tbody>${epochs.map(e=>`<tr><td>${esc(e.variant)} / ${e.epoch}</td><td>${e.train_loss.toFixed(4)}</td><td>${metricNumber(e.train.six_class_macro_f1)}</td><td>${metricNumber(e.validation.six_class_macro_f1)}</td><td>${metricNumber(e.validation.category_macro_f1)}</td></tr>`).join('')}</tbody></table></div>`:''}${models.length?`<h4>使用已保存的模型</h4><div class="training-models">${models.map((m,i)=>`<button class="button secondary small" data-trained-model="${i}">${esc(m.name)} → 帶入預測</button>`).join('')}</div>`:''}<details><summary>設定與最近事件（完整紀錄保存在本機）</summary><pre class="training-log">${esc(JSON.stringify({settings:job.config.kind==='bge-finetune'?job.config.settings:'使用既有訓練程式的固定設定',modal_settings:job.config.modal_settings,events:job.events},null,2))}</pre></details>`;
  document.querySelectorAll('[data-trained-model]').forEach(el=>el.addEventListener('click',()=>{const model=models[Number(el.dataset.trainedModel)];$('test-method').value=model.method;$('model-path').value=model.path;navigate('evaluate');$('experiment-form').scrollIntoView({behavior:'smooth',block:'start'});}));
  if(active)message('training-status','背景訓練中；切換頁面不會停止訓練。');
}
async function refreshTraining(preferred){
  clearTimeout(trainingTimer);
  try{
    const data=await api('/api/training');trainingJobs=data.jobs;$('start-training').disabled=trainingStarting||trainingJobs.some(j=>['queued','running'].includes(j.state));
    const selected=preferred||$('training-jobs').value||trainingJobs[0]?.job_id;
    $('training-jobs').innerHTML=trainingJobs.length?trainingJobs.map(j=>`<option value="${esc(j.job_id)}">${esc(j.job_id)} · ${esc(trainingNames[j.config.kind])} · ${esc(jobStates[j.state])}</option>`).join(''):'<option value="">尚無訓練紀錄</option>';
    if(trainingJobs.some(j=>j.job_id===selected))$('training-jobs').value=selected;
    const job=trainingJobs.find(j=>j.job_id===$('training-jobs').value);
    if(job)renderTraining(job);else{$('training-results').textContent='開始訓練後，這裡會顯示進度、比較結果與模型位置。';$('start-training').disabled=trainingStarting||trainingJobs.some(j=>['queued','running'].includes(j.state));}
    if(job&&!['queued','running'].includes(job.state))message('training-status',job.message,job.state!=='succeeded');
  }catch(error){message('training-status',error.message,true);}
  if(!$('page-training').hidden)trainingTimer=setTimeout(()=>refreshTraining(),2500);
}
$('open-training').addEventListener('click',()=>navigate('training'));
$('refresh-training').addEventListener('click',()=>refreshTraining());
$('training-jobs').addEventListener('change',()=>refreshTraining());
$('training-source').addEventListener('change',()=>{$('training-file-label').hidden=$('training-source').value!=='upload';});
$('training-file').addEventListener('change',()=>{$('training-file-label').textContent=$('training-file').files[0]?.name||'選擇 history JSONL';});
function trainingOptions(){
  const neural=$('training-kind').value==='bge-finetune';
  $('finetune-settings').hidden=!neural;$('finetune-settings').disabled=!neural;
  for(const option of $('training-backend').options)option.disabled=!neural&&option.value!=='cpu';
  if(!neural)$('training-backend').value='cpu';
  const backend=$('training-backend').value;
  $('training-cloud-label').hidden=backend!=='modal';$('training-cloud').required=backend==='modal';
  $('training-backend-hint').textContent=({cpu:'本機 CPU；既有 LR／SVC 使用 CPU。',gpu:'需要 NVIDIA 驅動與 CUDA 版 PyTorch；不可用時會回報失敗，不會偷偷切回 CPU。',modal:'GPU／CPU／RAM／timeout 讀取 Solution/config/modal_finetune.json。需先安裝 Modal SDK 並登入；只上傳本次選取的 history。'})[backend];
}
$('training-kind').addEventListener('change',trainingOptions);
$('training-backend').addEventListener('change',trainingOptions);
trainingOptions();
$('training-form').addEventListener('submit',async event=>{
  event.preventDefault();trainingStarting=true;$('start-training').disabled=true;
  try{
    const payload={kind:$('training-kind').value,backend:$('training-backend').value,cloud_upload_confirmed:$('training-cloud').checked,source:$('training-source').value,output_dir:$('training-output').value,history_only:$('training-history-only').checked};
    if(payload.kind==='bge-finetune')payload.settings={epochs:Number($('train-epochs').value),encoder_lr:Number($('train-encoder-lr').value),head_lr:Number($('train-head-lr').value),weight_decay:Number($('train-decay').value),dropout:Number($('train-dropout').value),label_smoothing:Number($('train-smoothing').value)};
    if(payload.source==='upload'){const file=$('training-file').files[0];if(!file)throw new Error('請選擇 history JSONL。');if(file.size>8*1024*1024)throw new Error('history 請小於 8 MB。');payload.history_name=file.name;payload.history_content=await file.text();}
    const job=await api('/api/training',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    message('training-status','已建立訓練工作；模型與紀錄會保存在時間戳資料夾。');await refreshTraining(job.job_id);
  }catch(error){message('training-status',error.message,true);}
  finally{trainingStarting=false;$('start-training').disabled=trainingJobs.some(j=>['queued','running'].includes(j.state));}
});

// Agent evaluations are independent persisted workers, so browser refreshes
// never reset the measured time or terminate a model evaluation.
let agentTimer=null, agentStarting=false, agentRowsKey='';
const agentKinds={history:'智慧問答 · history',synthetic:'智慧問答 · 合成測試',patrol:'主動巡檢 · 契約測試'};
const agentStates={queued:'等待中',running:'測試中',completed:'完成',failed:'執行失敗',interrupted:'已中斷'};
function agentSeconds(value){return typeof value==='number'?`${value.toFixed(3)} 秒`:'—';}
function renderAgent(job){
  $('agent-results').hidden=false;$('agent-empty').hidden=true;
  const active=['queued','running'].includes(job.state), rows=job.cases||[];
  const total=job.timing?.total_seconds;
  const maximum=rows.length?Math.max(...rows.map(row=>row.elapsed_s)):null;
  $('agent-run-title').textContent=`${agentKinds[job.kind]} · ${agentStates[job.state]}`;
  $('agent-progress').textContent=`已完成 ${job.completed} / ${job.total} · CPU ${job.threads} threads · 紀錄 ${job.job_id}`;
  $('agent-download').hidden=!job.report_url;if(job.report_url)$('agent-download').href=job.report_url;
  const cards=[['總耗時',agentSeconds(active?job.elapsed_seconds:total),active?'進行中，完成後保存精確總耗時':'含資料準備、逐題執行與報告儲存'],['通過題數',`${job.passed} / ${job.completed}`,job.kind==='patrol'?'契約檢查，不是排序準確率':`共 ${job.total} 題；未完成題目不列入分母`],['逐題耗時合計',agentSeconds(job.timing?.test_seconds),'每題含程序啟動與模型載入'],['最長單題',agentSeconds(maximum),job.kind==='patrol'?'巡檢上限 120 秒':'問答上限 30 秒']];
  $('agent-metrics').innerHTML=cards.map(([title,value,note])=>`<div class="metric-card"><div class="metric-name">${esc(title)}</div><div class="metric-value precise">${esc(value)}</div><div class="metric-desc">${esc(note)}</div></div>`).join('');
  const rowsKey=`${job.job_id}:${job.completed}:${job.state}`;
  if(rowsKey!==agentRowsKey){$('agent-rows').innerHTML=rows.length?rows.map(row=>{
    const actual=row.actual?.answer, brief=valueText(actual), error=row.error||(row.exit_code&&row.exit_code!==0?'程序未成功結束':row.elapsed_s>(job.kind==='patrol'?120:30)?'超過時間限制':!row.passed?'答案、型別或引用未通過檢查':'');
    return `<tr><td><div class="case-id">${esc(row.id)}</div>${esc(row.question||row.id)}<details><summary>完整輸出與執行紀錄</summary><pre>${esc(JSON.stringify({expected:row.expected,actual:row.actual,error:row.error,exit_code:row.exit_code,trace:row.trace},null,2))}</pre></details></td><td><span class="pill ${row.passed?'correct':''}">${row.passed?'通過':'未通過'}</span>${error?`<p class="detail-note">${esc(error)}</p>`:''}</td><td>${formatValue(row.expected)}</td><td>${esc(brief.length>180?brief.slice(0,180)+'…':brief)}</td><td class="case-time">${agentSeconds(row.elapsed_s)}</td></tr>`;
  }).join(''):'<tr><td colspan="5" class="muted">等待第一題完成；每題重新啟動並載入模型。</td></tr>';agentRowsKey=rowsKey;}
  message('agent-status',job.message,['failed','interrupted'].includes(job.state));
}
async function refreshAgent(preferred){
  clearTimeout(agentTimer);
  try{
    const data=await api('/api/agent/jobs');
    const selected=preferred||$('agent-jobs').value||data.jobs[0]?.job_id;
    $('agent-jobs').innerHTML=data.jobs.length?data.jobs.map(job=>`<option value="${esc(job.job_id)}">${esc(job.job_id)} · ${esc(agentKinds[job.kind])} · ${esc(agentStates[job.state])} · ${job.passed}/${job.completed}</option>`).join(''):'<option value="">尚無測試紀錄</option>';
    if(data.jobs.some(job=>job.job_id===selected))$('agent-jobs').value=selected;
    $('agent-start').disabled=agentStarting||!data.availability.ready||data.jobs.some(job=>['queued','running'].includes(job.state));
    $('agent-availability').textContent=data.availability.ready?'本機 Qwen3-1.7B Q4 與 CPU runtime 已就緒。測試期間不下載模型。':'缺少本機模型或 runtime。Windows 請先執行 python agent/setup.py；Linux/WSL 需安裝原生 CPU llama-server。';
    if($('agent-jobs').value)renderAgent(await api(`/api/agent/jobs?id=${encodeURIComponent($('agent-jobs').value)}`));
  }catch(error){message('agent-status',error.message,true);}
  if(!$('page-agent').hidden)agentTimer=setTimeout(()=>refreshAgent(),2000);
}
$('agent-form').addEventListener('submit',async event=>{
  event.preventDefault();agentStarting=true;$('agent-start').disabled=true;
  try{
    const job=await api('/api/agent/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:$('agent-kind').value,threads:Number($('agent-threads').value)})});
    await refreshAgent(job.job_id);
  }catch(error){message('agent-status',error.message,true);}
  finally{agentStarting=false;}
});
$('agent-refresh').addEventListener('click',()=>refreshAgent());
$('agent-jobs').addEventListener('change',()=>refreshAgent());
if(['agent','lookup','evaluate','training'].includes(location.hash.slice(1)))navigate(location.hash.slice(1));

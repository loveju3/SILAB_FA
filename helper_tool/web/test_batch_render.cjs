// 「方法分數比較」表：排名、最佳值、失敗／略過狀態與 HTML 跳脫；不需瀏覽器或資料檔。
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',hidden:false,value:'',options:[],listeners:{},
    classList:{toggle(){},add(){},remove(){}},addEventListener(name,fn){this.listeners[name]=fn;},
    querySelectorAll(){return[];},querySelector(){return element('option');}});
  return elements.get(id);
}
const context=vm.createContext({document:{getElementById:element,querySelectorAll:()=>[],querySelector:()=>element('container')},
  location:{hash:''},history:{replaceState(){}},setTimeout:()=>1,clearTimeout(){},console,
  fetch:async()=>({ok:true,json:async()=>({devices:[],sources:[],experiments:[],skipped:0,job:null})})});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'app.js'),'utf8'),context);
const ok=(method,total,seconds)=>({method,name:method,state:'succeeded',total,error_count:3,
  metrics:{category:total,device:.8,urgency:.7,duplicate:.6},timing:{total_seconds:seconds+1,prediction_seconds:seconds}});
(async()=>{
  await new Promise(setImmediate);
  assert.equal(element('batch-panel').hidden,true);
  context.job={job_id:'j',state:'completed',dataset:'history.jsonl',limit:'native',elapsed_seconds:12.5,items:[
    ok('baseline',.44,.2),{...ok('solution-a',.91,.5),model_label:'11/16 15:35 · TF-IDF <訓練>',model_path:'/runs/x'},{method:'tfidf-svc',name:'<b>svc</b>',state:'failed',error:'找不到 <模型>'},
    {method:'bge-finetuned',name:'ft',state:'skipped',error:'已略過'},ok('bge-lr',.7,3.25)]};
  vm.runInContext('renderBatch(job)',context);
  const html=element('batch-rows').innerHTML;
  assert.equal(element('batch-panel').hidden,false);
  assert.ok(html.indexOf('solution-a')<html.indexOf('bge-lr')&&html.indexOf('bge-lr')<html.indexOf('baseline'),'依總分排序');
  assert.ok(html.includes('#1')&&html.includes('#3'));
  assert.ok(html.includes('<td class="best">0.910000</td>'));
  assert.ok(html.includes('<td class="best">0.200</td>'),'最快預測標粗');
  assert.ok(html.includes('&lt;b&gt;svc&lt;/b&gt;')&&html.includes('找不到 &lt;模型&gt;'));
  assert.ok(html.includes('略過')&&html.includes('失敗'));
  assert.ok(html.includes('模型：11/16 15:35 · TF-IDF &lt;訓練&gt;'),'顯示並跳脫使用的模型');
  assert.ok(element('batch-progress').textContent.includes('最高分：solution-a（0.910000）'));
  context.job={job_id:'k',state:'running',dataset:'tickets.jsonl',limit:'2gb',elapsed_seconds:1,items:[
    {...ok('baseline',null,.1),total:null,metrics:null},{method:'solution-a',name:'A',state:'running'}]};
  vm.runInContext('renderBatch(job)',context);
  assert.ok(element('batch-rows').innerHTML.includes('未評分'));
  assert.ok(element('batch-progress').textContent.includes('目前：A'));
  assert.ok(element('batch-tag').textContent.includes('2 GB'));
  console.log('Batch rendering: ranking, best values, failures, unscored data and escaping passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});

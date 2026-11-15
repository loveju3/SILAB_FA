// Pure rendering contracts, with a tiny DOM double; no browser or network.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',hidden:false,value:'',options:[],
    classList:{toggle(name,value){this[name]=value;},add(){},remove(){}},
    addEventListener(){},querySelectorAll(){return [];},querySelector(){return element('option');}});
  return elements.get(id);
}
const context=vm.createContext({
  document:{getElementById:element,querySelectorAll:()=>[],querySelector:()=>element('container')},
  location:{hash:''},history:{replaceState(){}},setTimeout:()=>1,clearTimeout(){},
  fetch:async()=>({ok:true,json:async()=>({devices:[],sources:[]})}),console,
});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'app.js'),'utf8'),context);
const job={job_id:'synthetic-job',kind:'synthetic',state:'completed',threads:6,
  completed:1,total:30,passed:0,message:'測試完成',report_url:'/api/agent/report?id=synthetic-job',
  timing:{total_seconds:12.345,test_seconds:10.125},cases:[{id:'case-one',question:'<script>危險字串</script>',
  expected:4,rationale:'驗證 <b>交集</b> 條件',actual:{answer:0},passed:false,elapsed_s:10.125,exit_code:0,trace:'<img src=x>'}]};
context.job=job;
vm.runInContext('renderAgent(job)',context);
assert.equal(element('agent-results').hidden,false);
assert.equal(element('agent-download').hidden,false);
assert.ok(element('agent-metrics').innerHTML.includes('12.345 秒'));
assert.ok(element('agent-metrics').innerHTML.includes('共 30 題'));
assert.ok(element('agent-metrics').innerHTML.includes('10.125 秒'));
assert.ok(element('agent-rows').innerHTML.includes('未通過'));
assert.ok(element('agent-rows').innerHTML.includes('&lt;script&gt;'));
assert.ok(!element('agent-rows').innerHTML.includes('<script>'));
assert.ok(element('agent-rows').innerHTML.includes('&lt;img'));
assert.ok(element('agent-rows').innerHTML.includes('設計理由：'));
assert.ok(element('agent-rows').innerHTML.includes('驗證 &lt;b&gt;交集&lt;/b&gt; 條件'));
assert.ok(!element('agent-rows').innerHTML.includes('<b>交集</b>'));
// Polls must preserve an expanded details element when no new row arrived.
element('agent-rows').innerHTML='expanded-details-marker';
vm.runInContext('renderAgent(job)',context);
assert.equal(element('agent-rows').innerHTML,'expanded-details-marker');
context.job={...job,state:'running',elapsed_seconds:20.5,report_url:null};
vm.runInContext('renderAgent(job)',context);
assert.ok(element('agent-metrics').innerHTML.includes('20.500 秒'));
assert.equal(element('agent-download').hidden,true);
context.job={...job,state:'failed',message:'程序中斷',report_url:null};
vm.runInContext('renderAgent(job)',context);
assert.equal(element('agent-status').textContent,'程序中斷');
assert.equal(element('agent-status').classList.error,true);
context.job={...job,job_id:'old-report',cases:[{...job.cases[0],rationale:undefined}]};
vm.runInContext('renderAgent(job)',context);
assert.ok(element('agent-rows').innerHTML.includes('此紀錄未保存設計理由'));
console.log('Agent rendering contracts passed: timings, outcomes, escaping, details persistence and failure state.');
const patrolJob={...job,job_id:'patrol-real',kind:'patrol-tickets',passed:1,patrol:{
  source_info:{dataset:'tickets',file:'data/tickets.jsonl',count:2000,sha256:'snapshot-hash'},
  selected_tickets:[{ticket_id:'FAKE_A',priority:'high',reason:'原文依據',record_index:1750,source_line:1752,
    record:{ticket_id:'FAKE_A',reporter:'甲',created_at:'2026-10-01T00:00:00Z',raw_text:'<script>原文</script>\n換行'}},
    {ticket_id:'FAKE_B',priority:'medium',reason:'第二張',record_index:25,source_line:27,
    record:{ticket_id:'FAKE_B',reporter:'乙',created_at:'2026-10-01T00:00:00Z',raw_text:'第二張原文'}}]}};
context.job=patrolJob;
vm.runInContext('renderAgent(job)',context);
assert.equal(element('patrol-panel').hidden,false);
assert.ok(element('patrol-source').textContent.includes('2000'));
assert.ok(element('patrol-detail').innerHTML.includes('1750 / 2000'));
assert.ok(element('patrol-detail').innerHTML.includes('1752'));
assert.ok(element('patrol-detail').innerHTML.includes('&lt;script&gt;'));
assert.equal(element('patrol-prev').disabled,true);
vm.runInContext('selectPatrol(1)',context);
assert.ok(element('patrol-detail').innerHTML.includes('FAKE_B'));
assert.equal(element('patrol-next').disabled,true);
vm.runInContext('renderAgent(job)',context);
assert.ok(element('patrol-detail').innerHTML.includes('FAKE_B'));
context.job={...patrolJob,job_id:'legacy',kind:'patrol',patrol:undefined,cases:[{...job.cases[0],actual:{answer:[{ticket_id:'SYN001',priority:'high',reason:'合成'}]}}]};
vm.runInContext('renderAgent(job)',context);
assert.ok(element('patrol-source').textContent.includes('不對應正式'));
assert.ok(element('patrol-detail').innerHTML.includes('舊報告未保存'));
console.log('Patrol card contracts passed: original row/line, escaping, navigation, poll persistence and honest synthetic labels.');

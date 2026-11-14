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
  completed:1,total:12,passed:0,message:'測試完成',report_url:'/api/agent/report?id=synthetic-job',
  timing:{total_seconds:12.345,test_seconds:10.125},cases:[{id:'case-one',question:'<script>危險字串</script>',
  expected:4,actual:{answer:0},passed:false,elapsed_s:10.125,exit_code:0,trace:'<img src=x>'}]};
context.job=job;
vm.runInContext('renderAgent(job)',context);
assert.equal(element('agent-results').hidden,false);
assert.equal(element('agent-download').hidden,false);
assert.ok(element('agent-metrics').innerHTML.includes('12.345 秒'));
assert.ok(element('agent-metrics').innerHTML.includes('10.125 秒'));
assert.ok(element('agent-rows').innerHTML.includes('未通過'));
assert.ok(element('agent-rows').innerHTML.includes('&lt;script&gt;'));
assert.ok(!element('agent-rows').innerHTML.includes('<script>'));
assert.ok(element('agent-rows').innerHTML.includes('&lt;img'));
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
console.log('Agent rendering contracts passed: timings, outcomes, escaping, details persistence and failure state.');

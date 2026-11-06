#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>

static void hip_ok(hipError_t e, const char *call) { if(e!=hipSuccess) {std::fprintf(stderr,"HIP_FAIL %s %d %s\n",call,int(e),hipGetErrorString(e));throw std::runtime_error("HIP");} }
static void blas_ok(rocblas_status e, const char *call) { if(e!=rocblas_status_success) {std::fprintf(stderr,"ROCBLAS_FAIL %s %d\n",call,int(e));throw std::runtime_error("rocBLAS");} }
#define H(x) hip_ok((x),#x)
#define R(x) blas_ok((x),#x)
struct Card {int d=0;float *a=nullptr,*b=nullptr,*c=nullptr;hipStream_t stream=nullptr;rocblas_handle handle=nullptr;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;};
static bool cleanup(std::vector<Card>&cards) {
 bool ok=true;
 for(auto &c:cards) {
  if(hipSetDevice(c.d)!=hipSuccess){ok=false;continue;}
  if(c.stream && hipStreamSynchronize(c.stream)!=hipSuccess) {std::fprintf(stderr,"CLEANUP_DRAIN_FAIL device=%d allocations_retained_until_process_exit\n",c.d);ok=false;continue;}
  if(c.exec){if(hipGraphExecDestroy(c.exec)!=hipSuccess)ok=false;c.exec=nullptr;}
  if(c.graph){if(hipGraphDestroy(c.graph)!=hipSuccess)ok=false;c.graph=nullptr;}
  if(c.handle){if(rocblas_destroy_handle(c.handle)!=rocblas_status_success)ok=false;c.handle=nullptr;}
  for(float **p:{&c.a,&c.b,&c.c})if(*p){if(hipFree(*p)!=hipSuccess)ok=false;*p=nullptr;}
  if(c.stream){if(hipStreamDestroy(c.stream)!=hipSuccess)ok=false;c.stream=nullptr;}
 }
 return ok;
}
int main() {
 std::vector<Card> cards;int rc=0;
 try {
  int count=0;H(hipGetDeviceCount(&count));if(count!=8)throw std::runtime_error("require eight cards");
  constexpr int n=64;constexpr size_t elements=n*n,bytes=elements*sizeof(float);std::vector<float>a(elements),b(elements),expected(elements),host(elements);
  for(int col=0;col<n;++col)for(int row=0;row<n;++row){a[row+col*n]=float((row+2*col)%7-3);b[row+col*n]=float((3*row+col)%5-2);}
  for(int col=0;col<n;++col)for(int row=0;row<n;++row){float sum=0;for(int k=0;k<n;++k)sum+=a[row+k*n]*b[k+col*n];expected[row+col*n]=sum;}
  const float alpha=1.0f,beta=0.0f;
  cards.reserve(8);
  for(int d=0;d<8;++d){cards.emplace_back();Card &c=cards.back();c.d=d;H(hipSetDevice(d));H(hipStreamCreateWithFlags(&c.stream,hipStreamNonBlocking));H(hipMalloc((void**)&c.a,bytes));H(hipMalloc((void**)&c.b,bytes));H(hipMalloc((void**)&c.c,bytes));H(hipMemcpy(c.a,a.data(),bytes,hipMemcpyHostToDevice));H(hipMemcpy(c.b,b.data(),bytes,hipMemcpyHostToDevice));R(rocblas_create_handle(&c.handle));R(rocblas_set_stream(c.handle,c.stream));R(rocblas_set_pointer_mode(c.handle,rocblas_pointer_mode_host));}
  auto submit=[&](Card &c){H(hipSetDevice(c.d));R(rocblas_sgemm(c.handle,rocblas_operation_none,rocblas_operation_none,n,n,n,&alpha,c.a,n,c.b,n,&beta,c.c,n));};
  auto verify=[&](Card &c){H(hipSetDevice(c.d));H(hipStreamSynchronize(c.stream));H(hipMemcpy(host.data(),c.c,bytes,hipMemcpyDeviceToHost));for(size_t i=0;i<elements;++i)if(!std::isfinite(host[i])||host[i]!=expected[i]){std::fprintf(stderr,"VALUE_FAIL device=%d index=%zu got=%g expected=%g\n",c.d,i,host[i],expected[i]);throw std::runtime_error("numeric mismatch");}};
  for(int round=0;round<3;++round){for(auto &c:cards){H(hipSetDevice(c.d));H(hipMemsetAsync(c.c,0xff,bytes,c.stream));submit(c);}for(auto &c:cards){verify(c);std::printf("ROCBLAS_SGEMM_PASS round=%d device=%d n=%d checked=%zu\n",round,c.d,n,elements);}std::fflush(stdout);}
  for(auto &c:cards){H(hipSetDevice(c.d));H(hipStreamBeginCapture(c.stream,hipStreamCaptureModeGlobal));submit(c);H(hipStreamEndCapture(c.stream,&c.graph));H(hipGraphInstantiate(&c.exec,c.graph,nullptr,nullptr,0));}
  for(int round=0;round<3;++round){for(auto &c:cards){H(hipSetDevice(c.d));H(hipMemsetAsync(c.c,0xff,bytes,c.stream));H(hipGraphLaunch(c.exec,c.stream));}for(auto &c:cards){verify(c);std::printf("ROCBLAS_GRAPH_PASS round=%d device=%d checked=%zu\n",round,c.d,elements);}std::fflush(stdout);}
 }catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());rc=2;}
 if(!cleanup(cards))rc=10;
 if(rc==0)std::printf("PASS rocblas_numeric devices=8 sgemm_rounds=3 graph_rounds=3 exact_integer_fp32=true\n");
 return rc;
}

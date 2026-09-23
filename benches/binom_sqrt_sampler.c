// Single-thread CPU sampling without perf privileges. Only the profiling
// harness loads this library; production code and generated instructions stay
// unchanged. Resolve sampled PCs (including inline frames) with addr2line.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <ucontext.h>

#define CAPACITY (1 << 18)
static uintptr_t pcs[CAPACITY];
static volatile sig_atomic_t used;
static volatile sig_atomic_t lost;
static const char *output;

static void sample(int sig, siginfo_t *info, void *context) {
    (void)sig;
    (void)info;
    const ucontext_t *uc = context;
#if defined(__x86_64__)
    uintptr_t pc = uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
    uintptr_t pc = uc->uc_mcontext.pc;
#else
#error Unsupported sampling architecture
#endif
    if (used < CAPACITY)
        pcs[used++] = pc;
    else
        lost++;
}

__attribute__((constructor)) static void start(void) {
    output = getenv("BIGBITS_PROFILE_OUT");
    if (!output)
        return;
    struct sigaction action = {0};
    action.sa_sigaction = sample;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGPROF, &action, NULL) != 0) {
        perror("sigaction");
        abort();
    }
    struct itimerval timer = {{0, 1000}, {0, 1000}};
    if (setitimer(ITIMER_PROF, &timer, NULL) != 0) {
        perror("setitimer");
        abort();
    }
}

static int compare(const void *a, const void *b) {
    uintptr_t x = *(const uintptr_t *)a;
    uintptr_t y = *(const uintptr_t *)b;
    return (x > y) - (x < y);
}

__attribute__((destructor)) static void finish(void) {
    if (!output)
        return;
    sigset_t blocked;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPROF);
    sigprocmask(SIG_BLOCK, &blocked, NULL);
    struct itimerval timer = {0};
    setitimer(ITIMER_PROF, &timer, NULL);
    FILE *file = fopen(output, "w");
    if (!file) {
        perror("profile output");
        return;
    }
    fprintf(file, "# samples=%d lost=%d period_us=1000\n", used, lost);
    fprintf(file, "module\toffset\tcount\tsymbol\n");
    qsort(pcs, used, sizeof(pcs[0]), compare);
    for (int i = 0; i < used;) {
        int j = i + 1;
        while (j < used && pcs[j] == pcs[i])
            j++;
        Dl_info info = {0};
        dladdr((void *)pcs[i], &info);
        fprintf(file, "%s\t%lx\t%d\t%s\n",
                info.dli_fname ? info.dli_fname : "UNKNOWN",
                (unsigned long)(pcs[i] - (uintptr_t)info.dli_fbase), j - i,
                info.dli_sname ? info.dli_sname : "UNKNOWN");
        i = j;
    }
    fclose(file);
}

/* VocalSpike launcher: the executable inside VocalSpike.app.
 *
 * Starts the venv's Python on probe.py in one of three ways, so the tester can
 * see which one makes macOS attribute the child to VocalSpike instead of the
 * terminal:
 *   spawn     posix_spawn the child and wait (the launcher stays alive)
 *   disclaim  same, but the child disclaims responsibility (private API)
 *   exec      replace the launcher with Python
 * Mode comes from argv[1] (`open VocalSpike.app --args exec`) or
 * $VOCAL_SPIKE_MODE, default spawn. Further arguments go to probe.py.
 *
 * PYTHON_PATH, PROBE_PATH and LOG_PATH are baked in by build.sh.
 */
#include <dlfcn.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static void log_line(const char *fmt, const char *arg) {
    FILE *f = fopen(LOG_PATH, "a");
    if (!f) return;
    time_t now = time(NULL);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(f, "%s launcher[%d]: ", stamp, getpid());
    fprintf(f, fmt, arg);
    fputc('\n', f);
    fclose(f);
}

int main(int argc, char **argv) {
    const char *mode = getenv("VOCAL_SPIKE_MODE");
    int first_extra = 1;
    if (argc > 1 && (!strcmp(argv[1], "spawn") || !strcmp(argv[1], "disclaim") || !strcmp(argv[1], "exec"))) {
        mode = argv[1];
        first_extra = 2;
    }
    if (!mode) mode = "spawn";

    /* python probe.py --mode MODE [extra args...] NULL */
    int extras = argc - first_extra;
    char **child = calloc((size_t)extras + 5, sizeof(char *));
    child[0] = PYTHON_PATH;
    child[1] = PROBE_PATH;
    child[2] = "--mode";
    child[3] = (char *)mode;
    for (int i = 0; i < extras; i++) child[4 + i] = argv[first_extra + i];

    log_line("mode %s", mode);

    if (!strcmp(mode, "exec")) {
        execv(PYTHON_PATH, child);
        log_line("execv failed for %s", PYTHON_PATH);
        return 127;
    }

    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    if (!strcmp(mode, "disclaim")) {
        /* Private libSystem call used by terminals so their children own their TCC
         * grants; looked up at runtime so a missing symbol can't break the build. */
        int (*setdisclaim)(posix_spawnattr_t *, int) =
            (int (*)(posix_spawnattr_t *, int))dlsym(RTLD_DEFAULT, "responsibility_spawnattrs_setdisclaim");
        if (setdisclaim) {
            setdisclaim(&attr, 1);
        } else {
            log_line("%s", "responsibility_spawnattrs_setdisclaim unavailable; plain spawn");
        }
    }
    pid_t pid;
    int rc = posix_spawn(&pid, PYTHON_PATH, NULL, &attr, child, environ);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        log_line("posix_spawn failed: %s", strerror(rc));
        return 127;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

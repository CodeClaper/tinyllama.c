#include <sys/select.h>
#include <unistd.h>
#include "el.h"
#include "mm.h"

/* Create a file event. */
int create_file_event(EventLoop *el, int fd, int mask, elFileProc *proc, void *privdata) {
    FileEvent *fe;

    /* Construct a FileEvent. */
    fe = smalloc(sizeof(*fe));
    if (fe == NULL) return ELOOP_ERR;
    fe->fd = fd;
    fe->mask = mask;
    fe->proc = proc;
    fe->privdata = privdata;
    
    /* Append to fileEventHead. */
    if (el->fileEventHead) 
        fe->next = el->fileEventHead;
    el->fileEventHead = fe;

    return ELOOP_OK;
}

/* Delete a file event. */
void delete_file_event(EventLoop *el, int fd, int mask) {
    FileEvent *fe, *prev = NULL;
    fe = el->fileEventHead;
    while (fe != NULL) {
        if (fe->fd == fd && fe->mask == mask) {
            if (prev == NULL) 
                el->fileEventHead = fe->next;
            else
                prev->next = fe->next;
            sfree(fe);
            return;
        }
        prev = fe;
        fe = fe->next;
    }
}

/* Process event. */
void process_event(EventLoop *el, int flags) {
    int maxfd = 0, numfd = 0, numkeys = 0;
    fd_set rfds, wfds, efds;
    FileEvent *fe = el->fileEventHead, *next;

    /* If nothing, return back ASAP.*/
    if (!(flags & ELOOP_TIME_EVENTS) && !(flags & ELOOP_FILE_EVENTS)) return;

    FD_ZERO(&rfds);
    FD_ZERO(&wfds);
    FD_ZERO(&efds);

    if (flags & ELOOP_FILE_EVENTS) {
        while (fe != NULL) {
            if (fe->mask & ELOOP_READABLE) FD_SET(fe->fd, &rfds);
            if (fe->mask & ELOOP_WRITABLE) FD_SET(fe->fd, &wfds);
            if (fe->mask & ELOOP_EXCEPTION) FD_SET(fe->fd, &efds);
            if (maxfd < fe->fd) maxfd = fe->fd;
            numfd++;
            fe = fe->next;
        }
    }
    
    if (numfd > 0 || ((flags & ELOOP_TIME_EVENTS) && !(flags & ELOOP_DONT_WAIT))) {
        numkeys = select(maxfd + 1, &rfds, &wfds, &efds, NULL);
        if (numkeys > 0) {
            fe = el->fileEventHead;
            while (fe != NULL) {
                int fd = fe->fd;
                next = fe->next;
                if ((fe->mask & ELOOP_READABLE && FD_ISSET(fd, &rfds)) ||
                    (fe->mask & ELOOP_WRITABLE && FD_ISSET(fd, &wfds)) ||
                    (fe->mask & ELOOP_EXCEPTION && FD_ISSET(fd, &efds))
                ) {
                    int mask = 0;
                    if (fe->mask & ELOOP_READABLE && FD_ISSET(fd, &rfds)) mask |= ELOOP_READABLE;
                    if (fe->mask & ELOOP_WRITABLE && FD_ISSET(fd, &rfds)) mask |= ELOOP_WRITABLE;
                    if (fe->mask & ELOOP_EXCEPTION && FD_ISSET(fd, &rfds)) mask |= ELOOP_EXCEPTION;
                    
                    /* Procee file event. */
                    fe->proc(el, fd, mask, fe->privdata);

                    FD_CLR(fd, &rfds);
                    FD_CLR(fd, &wfds);
                    FD_CLR(fd, &efds);
                }
                fe = next;
            }
        }
    }

}

/* The main entry for event loop. */
void el_main(EventLoop *el) {
    while (!el->stop) {
        process_event(el, ELOOP_ALL_EVENTS);
    }
}

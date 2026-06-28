#include <asm-generic/errno-base.h>
#include <asm-generic/socket.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include "anet.h"
#include "utils.h"

/* Create TCP server. 
 * ------------------
 * Return the server socket fd.
 * Return ANET_ERR if fail.
 * */
int create_tcp_server(char *host, int port) {
    int s, on = 1;
    struct sockaddr_in sa;

    if (strcmp(host, "localhost") == 0) host = "127.0.0.1";
    /* Create server socket. */
    if ((s = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        fprintf(stderr, "Socket error: %s\n", strerror(errno));
        return ANET_ERR;
    }

    /* Set socket option. */
    if (setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on,  sizeof(on)) == -1) {
        fprintf(stderr, "Set socket SO_REUSEADDR error: %s\n", strerror(errno));
        close(s);
        return ANET_ERR;
    }

    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (!host || !host[0])
        sa.sin_addr.s_addr = htonl(INADDR_ANY);
    else 
        inet_pton(AF_INET, host, &sa.sin_addr);   

    /* Bind socket to port. */
    if (bind(s, (struct sockaddr *)&sa, sizeof(sa)) == -1) {
        fprintf(stderr, "Bind socket error: %s\n", strerror(errno));
        close(s);
        return ANET_ERR;
    }

    /* Listen to port. */
    if (listen(s, 10) == -1) {
        fprintf(stderr, "Listen error: %s\n", strerror(errno));
        close(s);
        return ANET_ERR;
    }

    return s;
}

/* Server socket accept client connection. */
int server_accept(int serversocket, char *clientIp, int *clientPort) {
    int fd;
    struct sockaddr_in sa;
    unsigned int salen;
    
    FOREVER {
        salen = sizeof(sa);
        fd = accept(serversocket, (struct sockaddr *)&sa, &salen);
        if (fd == -1) {
            if (errno == EINTR) continue;
            else {
                fprintf(stderr, "Accept error: %s\n", strerror(errno));
                return ANET_ERR;
            }
        }
        break;
    }

    if (clientIp) strcpy(clientIp, inet_ntoa(sa.sin_addr));
    if (clientPort) *clientPort = ntohs(sa.sin_port);
    
    return fd;
}

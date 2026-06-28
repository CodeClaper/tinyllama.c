#ifndef __ANET_H__
#define __ANET_H__

#define ANET_OK   1
#define ANET_ERR  0

int create_tcp_server(char *host, int port);
int server_accept(int serversocket, char *clientIp, int *clientPort);

#endif

#ifndef SOCKETS_HPP
#define SOCKETS_HPP

#include "utils/log.h"
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <arpa/inet.h>
#include <atomic>

/*
 * Creates, binds and registers then returns a listening file descriptor using
 * the provided `port`.
 */
int listener_socket(short port)
{
    /* Create a TCP/IPv4 listener socket. */
    int listener = socket(PF_INET, SOCK_STREAM, 0);
    if (listener < 0)
    {
        LOG.info("Error: Unable to create listener socket.\n");
        exit(-1);
    }

    /* Initialize the server IP address's structure. */
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    /*
     * Avoid bind() errors on re-use of Port.
     * NOTE: Does not terminate on failure, since it'd be a non-issue, right?
     */
    int v = 1;
    if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &v, sizeof(int)) < 0)
    {
        LOG.info("Failure: Unable to use setsockopt() to allow re-use of port.\n");
    }

    /* Bind listener socket to <IP,Port> pair. */
    if (bind(listener, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
        close(listener);
        LOG.info("Error: Unable to bind() listener socket to <IP,Port> pair.\n");
        exit(-1);
    }

    /*
     * Indicate passive (i.e., server) socket listening to client requests.
     * NOTE: I use FD_SETSIZE below. Some suggest using SOMAXCONN, which is
     * smaller on my system (128 vs 1024).
     */
    if (listen(listener, FD_SETSIZE) < 0)
    {
        close(listener);
        LOG.info("Error: Unable to listen() on socket.\n");
        exit(-1);
    }

    LOG.info("Created Listenning Socket\n");

    return listener;
}

#endif
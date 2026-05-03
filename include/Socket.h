#ifndef SOCKET_H
#define SOCKET_H
#include<sys/socket.h>
#include<unistd.h>
#include<stdio.h>


class Socket
{
public:
    Socket();
    explicit Socket(int fd);
    ~Socket();
    int getfd();
private:
    int fd;
};

#endif
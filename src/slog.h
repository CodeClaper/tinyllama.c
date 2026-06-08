
typedef enum { 
    INFO,       /* DB running Infomation. */
    SUCCESS,    /* Success result to client. */
    WARN,       /* For unexpected messages including sql syntaxt error, reapeated begin transaction or commit .etc. */ 
    ERROR,      /* User error, will abort transaction. */
} LogLevel;

void slog(LogLevel level, char *format, ...);
void slog_errno(char *format, ...);

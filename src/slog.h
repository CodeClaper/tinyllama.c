
typedef enum { 
    INFO,       /* DB running Infomation. */
    SUCCESS,    /* Success result to client. */
    WARN,       /* For unexpected messages including sql syntaxt error, reapeated begin transaction or commit .etc. */ 
    ERROR,      /* User error, will abort transaction. */
} LogLevel;

static char* LOG_LEVEL_NAME_LIST[] = { 
    "INFO", 
    "SUCCS", 
    "WARN", 
    "ERROR"
};

void slog(LogLevel level, char *format, ...);

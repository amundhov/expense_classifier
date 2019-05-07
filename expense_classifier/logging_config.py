config = { 
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
			"filename":"info.log"
        },
    },
    'loggers': { 
        '': { 
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        },
    } 
}

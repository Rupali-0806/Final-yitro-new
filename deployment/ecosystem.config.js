module.exports = {
  apps: [
    {
      name: 'yitro-crm',
      script: './dist/server/node-build.mjs',
      cwd: '/opt/yitro-crm',
      instances: 1,
      exec_mode: 'cluster',
      env: {
        NODE_ENV: 'production',
        PORT: 3000,
      },
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
      },
      error_file: '/var/log/yitro-crm/error.log',
      out_file: '/var/log/yitro-crm/out.log',
      log_file: '/var/log/yitro-crm/combined.log',
      time: true,
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      max_memory_restart: '1G',
      watch: false,
      ignore_watch: [
        'node_modules',
        'logs',
        '*.log'
      ],
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    }
  ]
};

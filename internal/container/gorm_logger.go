package container

import (
	"strings"
	"time"

	gormlogger "gorm.io/gorm/logger"
)

// newGormLogger keeps SQL observability without ever interpolating bound
// values. API keys, signed principals, document chunks and embeddings all
// pass through GORM and must not be copied into production logs.
func newGormLogger(writer gormlogger.Writer, requestedLevel string) gormlogger.Interface {
	level := gormlogger.Error
	switch strings.ToLower(strings.TrimSpace(requestedLevel)) {
	case "silent", "fatal":
		level = gormlogger.Silent
	case "warn", "warning":
		level = gormlogger.Warn
	case "error", "":
		level = gormlogger.Error
	}

	return gormlogger.New(writer, gormlogger.Config{
		SlowThreshold:             200 * time.Millisecond,
		LogLevel:                  level,
		IgnoreRecordNotFoundError: true,
		ParameterizedQueries:      true,
		Colorful:                  false,
	})
}

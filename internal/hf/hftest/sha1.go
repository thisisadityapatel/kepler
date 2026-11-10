package hftest

import (
	"crypto/sha1"
	"hash"
)

func sha1New() hash.Hash { return sha1.New() }
